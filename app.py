
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="FUB Building Energy Management System (BEMS)",
    page_icon="⚡",
    layout="wide",
)

# -------------------------------------------------------------------
# CONSTANTS & CONFIG
# -------------------------------------------------------------------
START_DATE = datetime(2025, 11, 1)
END_DATE = datetime(2025, 11, 15, 23, 45)  # last step
TIME_FREQ_MIN = 15  # minutes between data points

TARIFF_PER_KWH = 8.0       # BDT per kWh (demo)
CO2_PER_KWH_KG = 0.6       # kg CO2 per kWh (demo)

FLOORS = {
    "Floor 1": [
        ("F1-101", "Computer Lab 1"),
        ("F1-102", "Computer Lab 2"),
        ("F1-103", "Faculty Room 1"),
    ],
    "Floor 2": [
        ("F2-201", "Classroom 201"),
        ("F2-202", "Classroom 202"),
        ("F2-203", "Seminar Room"),
    ],
    "Floor 3": [
        ("F3-301", "Classroom 301"),
        ("F3-302", "Classroom 302"),
        ("F3-303", "Conference Room"),
    ],
}

# Demo weekly schedule for AC (when classes happen)
# Weekday (Mon–Fri): 9–12 and 13–17; Weekend (Sat–Sun): off
WEEKDAY_SCHEDULE = [
    (time(9, 0), time(12, 0)),
    (time(13, 0), time(17, 0)),
]

# -------------------------------------------------------------------
# STYLING
# -------------------------------------------------------------------
def inject_css():
    """Inject external CSS if available, otherwise fall back to inline."""
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        css = """
        <style>
        .metric-card {
            background: linear-gradient(135deg, #0f172a, #020617);
            border-radius: 18px;
            padding: 16px 20px;
            color: #e5e7eb;
            border: 1px solid rgba(148,163,184,0.35);
            box-shadow: 0 18px 40px rgba(15,23,42,0.55);
        }
        .metric-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9ca3af;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 1.7rem;
            font-weight: 700;
            color: #f9fafb;
        }
        .metric-sub {
            font-size: 0.78rem;
            color: #9ca3af;
            margin-top: 4px;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.3rem;
        }
        .small-caption {
            font-size: 0.8rem;
            color: #9ca3af;
            margin-bottom: 0.6rem;
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)


def metric_card(label, value, unit="", sub=None, emoji="⚡"):
    """Pretty metric card using raw HTML."""
    html = f"""
    <div class="metric-card">
        <div class="metric-label">{emoji} {label}</div>
        <div class="metric-value">{value}{(' ' + unit) if unit else ''}</div>
        <div class="metric-sub">{sub or ""}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# -------------------------------------------------------------------
# DATA GENERATION (SYNTHETIC, BUT REALISTIC-LOOKING)
# -------------------------------------------------------------------
def is_ac_on(ts: datetime) -> bool:
    """Return True if AC should be ON at this timestamp based on schedule."""
    weekday = ts.weekday()  # 0=Mon ... 6=Sun
    if weekday >= 5:  # Sat, Sun
        return False
    t = ts.time()
    for start_t, end_t in WEEKDAY_SCHEDULE:
        if start_t <= t <= end_t:
            return True
    return False


def generate_room_data(room_id: str, floor: str, room_name: str) -> pd.DataFrame:
    """Generate synthetic time-series for one room's smart device."""
    idx = pd.date_range(START_DATE, END_DATE, freq=f"{TIME_FREQ_MIN}min")
    n = len(idx)

    is_on = np.array([is_ac_on(ts) for ts in idx])

    # Base loads per room to make patterns slightly different
    base_current = np.random.uniform(7.0, 10.0)  # A
    base_voltage = np.random.uniform(225.0, 235.0)  # V

    voltage = np.where(
        is_on,
        np.random.normal(base_voltage, 2.0, size=n),
        np.random.normal(base_voltage - 2, 1.0, size=n),
    )

    current = np.where(
        is_on,
        np.random.normal(base_current, 0.8, size=n),
        np.random.normal(0.3, 0.1, size=n),  # standby current
    )

    power_kw = (voltage * current) / 1000.0  # kW
    power_kw = np.clip(power_kw, 0, None)

    step_hours = TIME_FREQ_MIN / 60.0
    energy_kwh_step = power_kw * step_hours

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "floor": floor,
            "room_id": room_id,
            "room_name": room_name,
            "voltage": voltage,
            "current": current,
            "power_kw": power_kw,
            "energy_kwh_step": energy_kwh_step,
            "is_on": is_on,
        }
    )
    return df


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    """Generate full 2-week dataset for all rooms."""
    frames = []
    for floor, rooms in FLOORS.items():
        for room_id, room_name in rooms:
            frames.append(generate_room_data(room_id, floor, room_name))
    df = pd.concat(frames, ignore_index=True)
    return df


# -------------------------------------------------------------------
# HELPERS FOR AGGREGATION
# -------------------------------------------------------------------
def filter_by_range(df: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    return df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)].copy()


def aggregate_energy(df: pd.DataFrame):
    total_energy = df["energy_kwh_step"].sum()
    total_cost = total_energy * TARIFF_PER_KWH
    total_co2_kg = total_energy * CO2_PER_KWH_KG
    return total_energy, total_cost, total_co2_kg


def daily_energy(df: pd.DataFrame):
    daily = (
        df.groupby(df["timestamp"].dt.date)["energy_kwh_step"]
        .sum()
        .reset_index(name="energy_kwh")
    )
    return daily


# -------------------------------------------------------------------
# UI SECTIONS
# -------------------------------------------------------------------
def sidebar_controls(df: pd.DataFrame):
    # Default view mode in session_state
    if "view_mode" not in st.session_state:
        st.session_state["view_mode"] = "Building Overview"

    st.sidebar.title("🏢 FUB BEMS")

    view = st.sidebar.radio(
        "View mode",
        ["Building Overview", "Floor View", "Room View", "Manage Devices (demo)"],
        key="view_mode",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Date & Time Range")

    start_default = START_DATE.date()
    end_default = END_DATE.date()

    date_range = st.sidebar.date_input(
        "Date range",
        (start_default, end_default),
        min_value=START_DATE.date(),
        max_value=END_DATE.date(),
    )

    if isinstance(date_range, (list, tuple)):
        start_date_sel, end_date_sel = date_range
    else:
        start_date_sel = end_date_sel = date_range

    start_time_sel = st.sidebar.time_input("Start time", value=time(8, 0))
    end_time_sel = st.sidebar.time_input("End time", value=time(20, 0))

    start_dt = datetime.combine(start_date_sel, start_time_sel)
    end_dt = datetime.combine(end_date_sel, end_time_sel)

    if end_dt <= start_dt:
        st.sidebar.error("End datetime must be after start datetime.")
    else:
        st.sidebar.caption(
            f"Showing data from **{start_dt}** to **{end_dt}**"
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Demo Settings")
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 60s (demo)", value=False)

    if auto_refresh:
        st.experimental_rerun()

    return view, start_dt, end_dt


def building_overview(df_range: pd.DataFrame):
    """Top-level dashboard for the whole FUB building, with clickable room tiles."""
    st.markdown("### 🏢 Building Overview")
    st.markdown(
        '<p class="small-caption">Central dashboard with clickable tiles for each room. '
        'Click a tile to open the detailed room dashboard.</p>',
        unsafe_allow_html=True,
    )

    total_energy, total_cost, total_co2_kg = aggregate_energy(df_range)

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card(
            "Total Energy",
            f"{total_energy:.1f}",
            "kWh",
            sub="Over selected time range",
            emoji="🔋",
        )
    with col2:
        metric_card(
            "Estimated Cost",
            f"{total_cost:,.0f}",
            "BDT",
            sub=f"Tariff {TARIFF_PER_KWH} BDT / kWh (demo)",
            emoji="💰",
        )
    with col3:
        metric_card(
            "CO₂ Emissions",
            f"{total_co2_kg * 1000:.0f}",
            "gCO₂",
            sub=f"{CO2_PER_KWH_KG} kg CO₂ / kWh (demo)",
            emoji="🌫️",
        )

    st.markdown("---")
    st.markdown('<div class="section-title">🏠 Clickable Room tiles (all floors)</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="small-caption">Each tile shows energy, bill and CO₂. '
        'Click any tile to jump to that room\'s detailed dashboard.</p>',
        unsafe_allow_html=True,
    )

    rooms_agg = (
        df_range
        .sort_values("timestamp")
        .groupby(["floor", "room_id", "room_name"])
        .agg(
            energy_kwh=("energy_kwh_step", "sum"),
            last_power_kw=("power_kw", "last"),
        )
        .reset_index()
        .sort_values(["floor", "room_id"])
    )

    if rooms_agg.empty:
        st.info("No rooms found in this range.")
    else:
        for i in range(0, len(rooms_agg), 3):
            cols = st.columns(3)
            sub_df = rooms_agg.iloc[i:i+3]
            for col, (_, row) in zip(cols, sub_df.iterrows()):
                with col:
                    cost = row["energy_kwh"] * TARIFF_PER_KWH
                    co2_g = row["energy_kwh"] * CO2_PER_KWH_KG * 1000
                    label = f"{row['floor']} · {row['room_id']}"
                    button_text = (
                        f"🏠 {label}\n"
                        f"{row['room_name']}\n"
                        f"{row['energy_kwh']:.1f} kWh | {cost:,.0f} BDT\n"
                        f"{co2_g:.0f} gCO₂ | Now: {row['last_power_kw']:.2f} kW"
                    )
                    if st.button(button_text, key=f"tile_{row['floor']}_{row['room_id']}"):
                        # Set session state so Room View opens with this room selected
                        st.session_state["floor_select"] = row["floor"]
                        room_label = f"{row['room_id']} - {row['room_name']}"
                        st.session_state["room_select_label"] = room_label
                        st.session_state["view_mode"] = "Room View"
                        st.experimental_rerun()

    st.markdown("---")
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown('<div class="section-title">📈 Building Load Profile</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="small-caption">Total power (kW) aggregated over all rooms.</p>',
            unsafe_allow_html=True,
        )
        power_series = (
            df_range.groupby("timestamp")["power_kw"].sum().reset_index()
        )
        power_series = power_series.rename(columns={"power_kw": "Total Power (kW)"})
        st.line_chart(
            power_series,
            x="timestamp",
            y="Total Power (kW)",
            use_container_width=True,
        )

    with col_b:
        st.markdown('<div class="section-title">🏬 Energy by Floor</div>', unsafe_allow_html=True)
        energy_by_floor = (
            df_range.groupby("floor")["energy_kwh_step"].sum().reset_index()
        )
        energy_by_floor = energy_by_floor.sort_values("energy_kwh_step", ascending=False)
        st.bar_chart(
            energy_by_floor.set_index("floor"),
            use_container_width=True,
        )
        st.caption("Floors with higher kWh indicate higher AC usage in that period.")

    st.markdown("---")
    st.markdown('<div class="section-title">📊 Daily Energy Trend (Building)</div>', unsafe_allow_html=True)
    daily = daily_energy(df_range)
    st.bar_chart(
        daily.set_index("timestamp"),
        use_container_width=True,
    )


def floor_view(df_range: pd.DataFrame):
    """Floor-level dashboard."""
    st.markdown("### 🏬 Floor View")

    # Default floor selection from session_state if available
    if "floor_select" not in st.session_state:
        st.session_state["floor_select"] = list(FLOORS.keys())[0]

    floor = st.selectbox("Select floor", list(FLOORS.keys()), key="floor_select")

    df_floor = df_range[df_range["floor"] == floor]

    if df_floor.empty:
        st.warning("No data for this floor in the selected range.")
        return

    total_energy, total_cost, total_co2_kg = aggregate_energy(df_floor)

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card(
            f"{floor} Energy",
            f"{total_energy:.1f}",
            "kWh",
            sub="Aggregated for selected range",
            emoji="🔌",
        )
    with col2:
        metric_card(
            "Estimated Cost",
            f"{total_cost:,.0f}",
            "BDT",
            sub="Floor-level cost (demo)",
            emoji="💵",
        )
    with col3:
        metric_card(
            "CO₂ Emissions",
            f"{total_co2_kg * 1000:.0f}",
            "gCO₂",
            sub="Floor-level footprint (demo)",
            emoji="🌍",
        )

    st.markdown("---")
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown('<div class="section-title">⚡ Power Profile (Floor)</div>', unsafe_allow_html=True)
        power_series = (
            df_floor.groupby("timestamp")["power_kw"].sum().reset_index()
        )
        power_series = power_series.rename(columns={"power_kw": "Power (kW)"})
        st.line_chart(
            power_series,
            x="timestamp",
            y="Power (kW)",
            use_container_width=True,
        )

    with col_b:
        st.markdown('<div class="section-title">🏠 Energy by Room</div>', unsafe_allow_html=True)
        energy_by_room = (
            df_floor.groupby(["room_id", "room_name"])["energy_kwh_step"]
            .sum()
            .reset_index()
        )
        energy_by_room["label"] = (
            energy_by_room["room_id"] + " - " + energy_by_room["room_name"]
        )
        st.bar_chart(
            energy_by_room.set_index("label")["energy_kwh_step"],
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown('<div class="section-title">📅 Daily Energy Trend (Floor)</div>', unsafe_allow_html=True)
    daily = daily_energy(df_floor)
    st.bar_chart(
        daily.set_index("timestamp"),
        use_container_width=True,
    )


def room_view(df_range: pd.DataFrame):
    """Room-level dashboard with real-time-style controls."""
    st.markdown("### 🏠 Room View")

    # Floor selection (controlled by session_state so tiles can override)
    if "floor_select" not in st.session_state:
        st.session_state["floor_select"] = list(FLOORS.keys())[0]
    floor = st.selectbox("Select floor", list(FLOORS.keys()), key="floor_select")

    room_options = FLOORS[floor]
    room_label_map = {
        f"{room_id} - {room_name}": room_id for room_id, room_name in room_options
    }

    # Ensure room_select_label exists and is valid for this floor
    if st.session_state.get("room_select_label") not in room_label_map:
        st.session_state["room_select_label"] = list(room_label_map.keys())[0]

    room_label = st.selectbox("Select room", list(room_label_map.keys()), key="room_select_label")
    room_id = room_label_map[room_label]

    df_room = df_range[df_range["room_id"] == room_id]

    if df_room.empty:
        st.warning("No data for this room in the selected range.")
        return

    room_name = df_room["room_name"].iloc[0]

    # Latest reading (simulate "real-time")
    df_room_sorted = df_room.sort_values("timestamp")
    latest = df_room_sorted.iloc[-1]

    total_energy, total_cost, total_co2_kg = aggregate_energy(df_room)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card(
            "Voltage (now)",
            f"{latest['voltage']:.1f}",
            "V",
            sub="Instantaneous reading",
            emoji="⚡",
        )
    with col2:
        metric_card(
            "Current (now)",
            f"{latest['current']:.2f}",
            "A",
            sub="Instantaneous reading",
            emoji="🔌",
        )
    with col3:
        metric_card(
            "Power (now)",
            f"{latest['power_kw']:.2f}",
            "kW",
            sub="Instantaneous reading",
            emoji="📡",
        )
    with col4:
        metric_card(
            "Energy (range)",
            f"{total_energy:.2f}",
            "kWh",
            sub="Total for selected period",
            emoji="🔋",
        )

    st.markdown(
        f'<p class="small-caption">Room: <strong>{room_id} – {room_name}</strong> | Floor: {floor}</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Voltage / Current / Power", "Daily Energy", "Controls & Schedule"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-title">📈 Voltage & Current</div>', unsafe_allow_html=True)
            vc_df = df_room_sorted[["timestamp", "voltage", "current"]].copy()
            vc_df = vc_df.set_index("timestamp")
            st.line_chart(vc_df, use_container_width=True)
        with col_b:
            st.markdown('<div class="section-title">⚡ Power Profile</div>', unsafe_allow_html=True)
            p_df = df_room_sorted[["timestamp", "power_kw"]].copy()
            p_df = p_df.set_index("timestamp")
            st.line_chart(p_df, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-title">📊 Daily Energy (kWh)</div>', unsafe_allow_html=True)
        daily = daily_energy(df_room_sorted)
        st.bar_chart(daily.set_index("timestamp"), use_container_width=True)

        st.markdown("#### 💰 Cost & CO₂ for this Room")
        col_c, col_d = st.columns(2)
        with col_c:
            st.write(f"**Estimated cost (range):** {total_cost:,.0f} BDT")
        with col_d:
            st.write(f"**Estimated emissions (range):** {total_co2_kg * 1000:.0f} gCO₂")

    with tab3:
        st.markdown('<div class="section-title">🧠 Smart Controls (Demo)</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="small-caption">Implements simple On/Off and speed control from the dashboard, '
            'plus a weekday schedule that can save ~20% energy when rooms are empty.</p>',
            unsafe_allow_html=True,
        )

        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            manual_on = st.toggle("Manual override: AC ON", value=bool(latest["is_on"]))
            fan_speed = st.slider("Fan speed / AC load (%)", 0, 100, 70, step=5)
            st.caption(
                "These controls simulate real-time On/Off and speed control. "
                "In a real deployment this would send commands to the IoT device."
            )

        with col_ctrl2:
            st.write("**Scheduled On/Off (Weekdays)**")
            schedule_df = pd.DataFrame(
                [
                    {"Mode": "Class time (Morning)", "From": "09:00", "To": "12:00"},
                    {"Mode": "Class time (Afternoon)", "From": "13:00", "To": "17:00"},
                ]
            )
            st.dataframe(schedule_df, use_container_width=True, hide_index=True)

        potential_savings = total_cost * 0.20
        st.markdown("---")
        st.write(
            f"💡 **Estimated savings with smart schedule:** "
            f"~{potential_savings:,.0f} BDT (≈20% of the bill for this room over the selected range)."
        )
        st.markdown(
            "> In your report, you can explain that automatically switching off AC when "
            "no one is in the room (outside class schedule) can realistically save 20–30% "
            "of electricity cost for cooling-dominated loads."
        )


def manage_devices_demo():
    """Simple CRUD-like interface for smart device IDs assigned to rooms."""
    st.markdown("### 🛠 Manage Devices (Demo Only – In-Memory)")
    st.markdown(
        '<p class="small-caption">Add / delete / update smart device IDs and assign them to rooms. '
        'This mimics a real BEMS device registry.</p>',
        unsafe_allow_html=True,
    )

    if "devices_df" not in st.session_state:
        rows = []
        for floor, rooms in FLOORS.items():
            for room_id, room_name in rooms:
                rows.append(
                    {
                        "floor": floor,
                        "room_id": room_id,
                        "room_name": room_name,
                        "iot_device_id": f"{room_id}-AC",
                        "status": "active",
                    }
                )
        st.session_state.devices_df = pd.DataFrame(rows)

    edited = st.data_editor(
        st.session_state.devices_df,
        num_rows="dynamic",
        use_container_width=True,
        key="devices_editor",
    )
    st.session_state.devices_df = edited

    st.markdown("#### JSON config preview (for documentation / export)")
    st.json(edited.to_dict(orient="records"))


# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------
def main():
    inject_css()

    st.title("⚡ FUB Building Energy Management System (BEMS)")
    st.caption(
        "IoT-based real-time energy monitoring & control • Demo dashboard for CSE407 – Green Computing"
    )

    df = load_data()

    view, start_dt, end_dt = sidebar_controls(df)

    if end_dt <= start_dt:
        st.stop()

    df_range = filter_by_range(df, start_dt, end_dt)

    if df_range.empty:
        st.warning("No data in the selected range. Try expanding the date/time.")
        st.stop()

    if view == "Building Overview":
        building_overview(df_range)
    elif view == "Floor View":
        floor_view(df_range)
    elif view == "Room View":
        room_view(df_range)
    else:
        manage_devices_demo()


if __name__ == "__main__":
    main()
