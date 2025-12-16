import streamlit as st
import pandas as pd
import sqlite3
import uuid
from datetime import date, datetime, timedelta
import time
import plotly.express as px
import plotly.graph_objects as go

# ---------- CONFIG ----------
st.set_page_config(page_title="Corporate Portal", layout="wide", page_icon="🏢")

# ---------- STYLES ----------
st.markdown(
    """
    <style>
    .stApp { background-color: #f8f9fa; }
    
    /* Profile Image */
    .profile-img {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        border: 4px solid #dfe6e9;
        display: block;
        margin: 0 auto 15px auto;
    }
    
    /* Clean Container Styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Button Tweak */
    div.stButton > button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
    }
    
    /* Plotly Transparent Background */
    .js-plotly-plot .plotly .main-svg {
        background-color: rgba(0,0,0,0) !important;
    }
    
    /* Status Badges */
    .status-ok { color: #10b981; font-weight: bold; }
    .status-pending { color: #f59e0b; font-weight: bold; }
    .status-block { color: #ef4444; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- DATABASE ----------
DB_FILE = "portal_data_final_v14_demo.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # 1. KPI Table
    c.execute('''CREATE TABLE IF NOT EXISTS tasks_v2 (
        id TEXT PRIMARY KEY, name_activity_pilot TEXT, task_name TEXT, date_of_receipt TEXT,
        actual_delivery_date TEXT, commitment_date_to_customer TEXT, status TEXT,
        ftr_customer TEXT, reference_part_number TEXT, ftr_internal TEXT, otd_internal TEXT,
        description_of_activity TEXT, activity_type TEXT, ftr_quality_gate_internal TEXT,
        date_of_clarity_in_input TEXT, start_date TEXT, otd_customer TEXT, customer_remarks TEXT,
        name_quality_gate_referent TEXT, project_lead TEXT, customer_manager_name TEXT
    )''')
    
    # 2. Training Tables
    c.execute('''CREATE TABLE IF NOT EXISTS training_repo (
        id TEXT PRIMARY KEY, title TEXT, description TEXT, link TEXT, 
        role_target TEXT, mandatory INTEGER, created_by TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS training_progress (
        user_name TEXT, training_id TEXT, status TEXT, 
        last_updated TEXT, PRIMARY KEY (user_name, training_id)
    )''')
    
    # 3. Resource Tracker Table
    c.execute('''CREATE TABLE IF NOT EXISTS onboarding_details (
        username TEXT PRIMARY KEY,
        fullname TEXT,
        emp_id TEXT,
        tid TEXT,
        location TEXT,
        work_mode TEXT,
        hr_policy_briefing INTEGER,
        it_system_setup INTEGER,
        tid_active INTEGER,
        team_centre_training INTEGER,
        agt_access INTEGER,
        ext_mail_id INTEGER,
        rdp_access INTEGER,
        avd_access INTEGER,
        teamcenter_access INTEGER,
        blocking_point TEXT,
        ticket_raised TEXT
    )''')
    
    # --- DEMO DATA INJECTION ---
    
    # Check & Add Training Demo Data
    c.execute("SELECT count(*) FROM training_repo")
    if c.fetchone()[0] == 0:
        trainings = [
            ("TR-01", "Python Basics", "Introduction to Python syntax", "https://python.org", "All", 1, "System"),
            ("TR-02", "Advanced Pandas", "Data manipulation mastery", "https://pandas.pydata.org", "Team Member", 0, "System"),
            ("TR-03", "Streamlit UI", "Building interactive dashboards", "https://streamlit.io", "All", 1, "System"),
            ("TR-04", "Workplace Safety", "Fire & Health safety protocols", "https://osha.gov", "All", 1, "System"),
            ("TR-05", "Leadership 101", "Managing high-performance teams", "https://hbr.org", "Team Leader", 1, "System"),
            ("TR-06", "Agile Scrum", "Sprints, standups and retrospectives", "https://scrum.org", "All", 0, "System"),
            ("TR-07", "Git Version Control", "Branching strategies and PRs", "https://github.com", "Team Member", 1, "System"),
            ("TR-08", "Cyber Security", "Phishing awareness and data privacy", "https://security.com", "All", 1, "System"),
            ("TR-09", "Conflict Resolution", "HR guidelines for conflicts", "https://hr.com", "Team Leader", 0, "System"),
            ("TR-10", "Cloud Computing", "AWS Fundamentals", "https://aws.amazon.com", "Team Member", 0, "System")
        ]
        c.executemany("INSERT INTO training_repo VALUES (?,?,?,?,?,?,?)", trainings)

    # Check & Add Resource Demo Data
    c.execute("SELECT count(*) FROM onboarding_details")
    if c.fetchone()[0] == 0:
        resources = [
            ("res1", "Alice Johnson", "EMP-201", "TID-201", "Chennai", "Office", 1, 1, 1, 1, 0, 1, 0, 0, 1, "", ""),
            ("res2", "Bob Smith", "EMP-202", "TID-202", "Pune", "Remote", 1, 1, 0, 0, 1, 0, 1, 1, 0, "VPN Issue", "TKT-1029"),
            ("res3", "Charlie Brown", "EMP-203", "TID-203", "Bangalore", "Office", 1, 1, 1, 1, 0, 1, 0, 0, 1, "", ""),
            ("res4", "Diana Prince", "EMP-204", "TID-204", "Chennai", "Remote", 0, 0, 0, 0, 0, 0, 0, 0, 0, "", ""),
            ("res5", "Evan Wright", "EMP-205", "TID-205", "Pune", "Office", 1, 1, 1, 0, 0, 1, 0, 0, 0, "Laptop Delay", "TKT-3321"),
            ("res6", "Fiona Gallagher", "EMP-206", "TID-206", "Client Site", "Office", 1, 1, 1, 1, 0, 1, 0, 0, 1, "", ""),
            ("res7", "George Martin", "EMP-207", "TID-207", "Bangalore", "Remote", 1, 0, 0, 0, 1, 0, 1, 0, 0, "", ""),
            ("res8", "Hannah Baker", "EMP-208", "TID-208", "Chennai", "Office", 1, 1, 1, 1, 0, 1, 0, 0, 1, "", ""),
            ("res9", "Ian Somerhalder", "EMP-209", "TID-209", "Pune", "Remote", 1, 1, 1, 1, 1, 1, 1, 1, 1, "", ""),
            ("res10", "Jack Daniels", "EMP-210", "TID-210", "Chennai", "Office", 0, 1, 0, 0, 0, 0, 0, 0, 0, "No ID Card", "")
        ]
        c.executemany("INSERT INTO onboarding_details VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", resources)

    conn.commit()
    conn.close()

# ---------- UTILS & HELPERS ----------

# --- KPI HELPERS ---
def get_kpi_data():
    conn = sqlite3.connect(DB_FILE)
    try: df = pd.read_sql_query("SELECT * FROM tasks_v2", conn)
    except: df = pd.DataFrame()
    conn.close()
    return df

def save_kpi_task(data, task_id=None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    otd_val = "N/A"
    try:
        ad = data.get("actual_delivery_date")
        cd = data.get("commitment_date_to_customer")
        if ad and cd and ad != 'None' and cd != 'None':
            a_dt = pd.to_datetime(ad, dayfirst=True, errors='coerce')
            c_dt = pd.to_datetime(cd, dayfirst=True, errors='coerce')
            if not pd.isna(a_dt) and not pd.isna(c_dt):
                otd_val = "OK" if a_dt <= c_dt else "NOT OK"
    except: pass

    cols = ['name_activity_pilot', 'task_name', 'date_of_receipt', 'actual_delivery_date', 
            'commitment_date_to_customer', 'status', 'ftr_customer', 'reference_part_number', 
            'ftr_internal', 'otd_internal', 'description_of_activity', 'activity_type', 
            'ftr_quality_gate_internal', 'date_of_clarity_in_input', 'start_date', 'otd_customer', 
            'customer_remarks', 'name_quality_gate_referent', 'project_lead', 'customer_manager_name']
    
    data['otd_internal'] = otd_val; data['otd_customer'] = otd_val
    vals = [str(data.get(k, '')) if data.get(k) is not None else '' for k in cols]

    if task_id:
        set_clause = ", ".join([f"{col}=?" for col in cols])
        c.execute(f"UPDATE tasks_v2 SET {set_clause} WHERE id=?", (*vals, task_id))
    else:
        new_id = str(uuid.uuid4())[:8]
        placeholders = ",".join(["?"] * (len(cols) + 1))
        c.execute(f"INSERT INTO tasks_v2 VALUES ({placeholders})", (new_id, *vals))
    conn.commit(); conn.close()

def import_kpi_csv(file):
    try:
        df = pd.read_csv(file)
        if 'id' not in df.columns: df['id'] = [str(uuid.uuid4())[:8] for _ in range(len(df))]
        
        required_cols = ["name_activity_pilot", "task_name", "date_of_receipt", "actual_delivery_date",
            "commitment_date_to_customer", "status", "ftr_customer", "reference_part_number",
            "ftr_internal", "otd_internal", "description_of_activity", "activity_type",
            "ftr_quality_gate_internal", "date_of_clarity_in_input", "start_date", "otd_customer",
            "customer_remarks", "name_quality_gate_referent", "project_lead", "customer_manager_name"]
            
        for col in required_cols: 
            if col not in df.columns: df[col] = None
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        cols_to_keep = ['id'] + required_cols
        df = df[cols_to_keep]
        for index, row in df.iterrows():
            placeholders = ','.join(['?'] * len(row))
            sql = f"INSERT OR REPLACE INTO tasks_v2 VALUES ({placeholders})"
            c.execute(sql, tuple(row))
        conn.commit(); conn.close()
        return True
    except: return False

# --- TRAINING HELPERS ---
def add_training(title, desc, link, role, mandatory, creator):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    tid = str(uuid.uuid4())[:8]
    c.execute("INSERT INTO training_repo VALUES (?,?,?,?,?,?,?)", 
              (tid, title, desc, link, role, 1 if mandatory else 0, creator))
    conn.commit(); conn.close()

def get_trainings(user_name=None):
    conn = sqlite3.connect(DB_FILE)
    repo = pd.read_sql_query("SELECT * FROM training_repo", conn)
    if user_name:
        prog = pd.read_sql_query("SELECT * FROM training_progress WHERE user_name=?", conn, params=(user_name,))
        if not repo.empty:
            merged = pd.merge(repo, prog, left_on='id', right_on='training_id', how='left')
            merged['status'] = merged['status'].fillna('Not Started')
            conn.close(); return merged
    conn.close(); return repo

def update_training_status(user_name, training_id, status):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO training_progress VALUES (?,?,?,?)", 
              (user_name, training_id, status, str(date.today())))
    conn.commit(); conn.close()

def import_training_csv(file):
    try:
        df = pd.read_csv(file)
        if 'title' not in df.columns: return False
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        for _, row in df.iterrows():
            tid = str(uuid.uuid4())[:8]
            c.execute("INSERT INTO training_repo VALUES (?,?,?,?,?,?,?)", 
                      (tid, row['title'], row.get('description',''), row.get('link',''), 
                       row.get('role_target','All'), int(row.get('mandatory',0)), st.session_state['name']))
        conn.commit(); conn.close()
        return True
    except: return False

# --- RESOURCE TRACKER (ONBOARDING) HELPERS ---
def get_onboarding_details(username):
    conn = sqlite3.connect(DB_FILE)
    try: df = pd.read_sql_query("SELECT * FROM onboarding_details WHERE username=?", conn, params=(username,))
    except: df = pd.DataFrame()
    conn.close()
    return df

def get_all_onboarding():
    conn = sqlite3.connect(DB_FILE)
    try: df = pd.read_sql_query("SELECT * FROM onboarding_details", conn)
    except: df = pd.DataFrame()
    conn.close()
    return df

def save_onboarding_details(data):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT username FROM onboarding_details WHERE username=?", (data['username'],))
    exists = c.fetchone()
    cols = ['username', 'fullname', 'emp_id', 'tid', 'location', 'work_mode',
            'hr_policy_briefing', 'it_system_setup', 'tid_active', 'team_centre_training',
            'agt_access', 'ext_mail_id', 'rdp_access', 'avd_access', 'teamcenter_access',
            'blocking_point', 'ticket_raised']
    vals = [data.get(k) for k in cols]
    if exists:
        set_clause = ", ".join([f"{col}=?" for col in cols])
        c.execute(f"UPDATE onboarding_details SET {set_clause} WHERE username=?", (*vals, data['username']))
    else:
        placeholders = ",".join(["?"] * len(cols))
        c.execute(f"INSERT INTO onboarding_details VALUES ({placeholders})", vals)
    conn.commit(); conn.close()

# --- PLOTLY HELPERS ---
def get_analytics_chart(df):
    if df.empty: return go.Figure()
    df_local = df.copy()
    df_local['start_date'] = pd.to_datetime(df_local['start_date'], dayfirst=True, errors='coerce')
    df_local = df_local.dropna(subset=['start_date'])
    if df_local.empty: return go.Figure()
    df_local['month'] = df_local['start_date'].dt.strftime('%b')
    monthly = df_local.groupby(['month', 'status']).size().reset_index(name='count')
    fig = px.bar(monthly, x='month', y='count', color='status', barmode='group',
                 color_discrete_map={"Completed":"#10b981","Inprogress":"#3b82f6","Hold":"#ef4444","Cancelled":"#9ca3af"})
    fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=300)
    return fig

def get_donut(df):
    if df.empty: return go.Figure()
    total = len(df)
    completed = len(df[df['status']=='Completed'])
    completed_pct = int((completed/total)*100) if total>0 else 0
    fig = go.Figure(data=[go.Pie(labels=['Completed','Pending'], values=[completed_pct, 100-completed_pct], hole=.7, textinfo='none')])
    fig.update_layout(height=240, margin=dict(l=0,r=0,t=0,b=0), 
                      annotations=[dict(text=f"{completed_pct}%", x=0.5, y=0.5, showarrow=False, font=dict(size=20))])
    return fig

# ---------- AUTH ----------
USERS = {
    "leader": {"password": "123", "role": "Team Leader", "name": "Sarah Jenkins", "emp_id": "LDR-001", "tid": "TID-999", "img": "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?auto=format&fit=crop&q=80&w=200&h=200"},
    "member1": {"password": "123", "role": "Team Member", "name": "David Chen", "emp_id": "EMP-101", "tid": "TID-101", "img": "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?auto=format&fit=crop&q=80&w=200&h=200"},
    "member2": {"password": "123", "role": "Team Member", "name": "Emily Davis", "emp_id": "EMP-102", "tid": "TID-102", "img": "https://images.unsplash.com/photo-1580489944761-15a19d654956?auto=format&fit=crop&q=80&w=200&h=200"}
}

def login_page():
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        with st.container(border=True):
            st.markdown("<h2 style='text-align:center; color:#1f2937;'>Portal Sign In</h2>", unsafe_allow_html=True)
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Secure Login", use_container_width=True, type="primary"):
                if u in USERS and USERS[u]["password"] == p:
                    st.session_state.update({
                        'logged_in': True, 'user': u, 'role': USERS[u]['role'], 
                        'name': USERS[u]['name'], 'emp_id': USERS[u].get('emp_id'),
                        'tid': USERS[u].get('tid'), 'img': USERS[u]['img'], 'current_app': 'HOME'
                    })
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

# ---------- APP SECTIONS ----------
def app_home():
    st.markdown(f"## Welcome, {st.session_state['name']}")
    st.caption(f"ID: {st.session_state.get('emp_id')} | TID: {st.session_state.get('tid')}")
    st.write("---")
    
    # 1 LINE OF 4 CARDS (RESTORED TO DESKTOP DEFAULT)
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        with st.container(border=True):
            st.markdown("### 📊 **KPI System**"); st.caption("Manage OTD & FTR")
            if st.button("Launch KPI", use_container_width=True, type="primary"): st.session_state['current_app']='KPI'; st.rerun()
    with c2:
        with st.container(border=True):
            st.markdown("### 🎓 **Training**"); st.caption("Track Progress")
            if st.button("Launch Training", use_container_width=True, type="primary"): st.session_state['current_app']='TRAINING'; st.rerun()
    with c3:
        with st.container(border=True):
            st.markdown("### 🚀 **Resource Tracker**"); st.caption("Onboarding & Info")
            if st.button("Launch Tracker", use_container_width=True, type="primary"): st.session_state['current_app']='RESOURCE'; st.rerun()
    with c4:
        with st.container(border=True):
            st.markdown("### 🕸️ **Skill Radar**"); st.caption("Team Matrix")
            if st.button("View Radar", use_container_width=True): st.toast("🚧 Under Construction!", icon="👷")

def parse_date(d):
    if not d or d == 'None': return None
    try: return pd.to_datetime(d).date()
    except: return None

# --- FULL KPI APP ---
def app_kpi():
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("⬅ Home", use_container_width=True): st.session_state['current_app']='HOME'; st.rerun()
    with c2: st.markdown("### 📊 KPI Management System")
    st.markdown("---")
    
    # TEAM LEADER
    if st.session_state['role'] == "Team Leader":
        df = get_kpi_data()
        
        # 1. Editor
        if 'edit_kpi_id' not in st.session_state: st.session_state['edit_kpi_id'] = None
        if st.session_state['edit_kpi_id']:
            with st.container(border=True):
                is_new = st.session_state['edit_kpi_id'] == 'NEW'
                st.subheader("Create/Edit Task")
                default_data = {}
                if not is_new:
                    task_row = df[df['id'] == st.session_state['edit_kpi_id']]
                    if not task_row.empty: default_data = task_row.iloc[0].to_dict()

                with st.form("kpi_editor_form"):
                    c1, c2, c3 = st.columns(3)
                    pilots = [u['name'] for k,u in USERS.items() if u['role']=="Team Member"]
                    with c1:
                        tname = st.text_input("Task Name", value=default_data.get("task_name", ""))
                        pilot_val = default_data.get("name_activity_pilot")
                        p_idx = pilots.index(pilot_val) if pilot_val in pilots else 0
                        pilot = st.selectbox("Assign To", pilots, index=p_idx)
                    with c2:
                        statuses = ["Hold", "Inprogress", "Completed", "Cancelled"]
                        stat_val = default_data.get("status", "Inprogress")
                        s_idx = statuses.index(stat_val) if stat_val in statuses else 1
                        status = st.selectbox("Status", statuses, index=s_idx)
                        start_d = st.date_input("Start Date", value=parse_date(default_data.get("start_date")) or date.today())
                    with c3:
                        comm_d = st.date_input("Commitment Date", value=parse_date(default_data.get("commitment_date_to_customer")) or date.today()+timedelta(days=7))
                        act_d = st.date_input("Actual Delivery", value=parse_date(default_data.get("actual_delivery_date")) or date.today())
                    st.divider()
                    c4, c5 = st.columns(2)
                    with c4:
                        desc = st.text_area("Description", value=default_data.get("description_of_activity", ""))
                        ref_part = st.text_input("Ref Part #", value=default_data.get("reference_part_number", ""))
                    with c5:
                        ftr = st.selectbox("FTR Internal", ["Yes", "No"], index=0)
                        rem = st.text_area("Remarks", value=default_data.get("customer_remarks", ""))
                    
                    if st.form_submit_button("💾 Save Task", type="primary", use_container_width=True):
                        payload = {
                            "task_name": tname, "name_activity_pilot": pilot, "status": status,
                            "start_date": str(start_d), "commitment_date_to_customer": str(comm_d),
                            "actual_delivery_date": str(act_d), "description_of_activity": desc,
                            "reference_part_number": ref_part, "ftr_internal": ftr, "customer_remarks": rem,
                            "date_of_receipt": str(date.today()), "activity_type": "Standard"
                        }
                        save_kpi_task(payload, None if is_new else st.session_state['edit_kpi_id'])
                        st.success("Saved successfully!")
                        st.session_state['edit_kpi_id'] = None
                        st.rerun()
                if st.button("Cancel", use_container_width=True):
                    st.session_state['edit_kpi_id'] = None; st.rerun()
            st.markdown("---")

        # 2. Dashboard
        if not st.session_state['edit_kpi_id']:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Tasks", len(df))
            m2.metric("In Progress", len(df[df['status']=='Inprogress']) if not df.empty else 0)
            m3.metric("On Hold", len(df[df['status']=='Hold']) if not df.empty else 0)
            m4.metric("Completed", len(df[df['status']=='Completed']) if not df.empty else 0)
            
            tb1, tb2 = st.columns([3, 1])
            with tb1:
                with st.expander("📂 CSV Import/Export"):
                    up = st.file_uploader("Import CSV", type=['csv'])
                    if up: 
                        if import_kpi_csv(up): st.success("Imported!"); st.rerun()
                    if not df.empty:
                        st.download_button("Export CSV", data=df.to_csv(index=False).encode('utf-8'), file_name="kpi.csv", mime="text/csv")
            with tb2:
                if st.button("➕ New Task", type="primary", use_container_width=True):
                    st.session_state['edit_kpi_id'] = "NEW"; st.rerun()

            c_chart, c_donut = st.columns([2, 1])
            if not df.empty:
                with c_chart: st.plotly_chart(get_analytics_chart(df), use_container_width=True)
                with c_donut: st.plotly_chart(get_donut(df), use_container_width=True)
            
            st.markdown("#### Active Tasks")
            if not df.empty:
                for idx, row in df.iterrows():
                    with st.container(border=True):
                        c_main, c_meta, c_btn = st.columns([4, 2, 1])
                        with c_main:
                            st.markdown(f"**{row['task_name']}**")
                            st.caption(row.get('description_of_activity',''))
                        with c_meta:
                            st.caption(f"👤 {row.get('name_activity_pilot','-')}")
                            st.caption(f"📅 Due: {row.get('commitment_date_to_customer','-')}")
                            st.write(f"**{row['status']}**")
                        with c_btn:
                            if st.button("Edit", key=f"kpi_edit_{row['id']}", use_container_width=True):
                                st.session_state['edit_kpi_id'] = row['id']; st.rerun()
            else: st.info("No tasks found.")

    # MEMBER
    else:
        df = get_kpi_data()
        my_tasks = df[df['name_activity_pilot'] == st.session_state['name']]
        st.metric("My Pending Tasks", len(my_tasks[my_tasks['status']!='Completed']) if not my_tasks.empty else 0)
        if not my_tasks.empty:
            for idx, row in my_tasks.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['task_name']}**")
                    st.write(f"Due: {row.get('commitment_date_to_customer','-')}")
                    with st.form(key=f"my_task_{row['id']}"):
                        c1, c2 = st.columns(2)
                        curr_stat = row.get('status', 'Inprogress')
                        idx_stat = ["Inprogress", "Completed", "Hold"].index(curr_stat) if curr_stat in ["Inprogress", "Completed", "Hold"] else 0
                        ns = c1.selectbox("Status", ["Inprogress", "Completed", "Hold"], index=idx_stat)
                        ad = c2.date_input("Actual Delivery", value=parse_date(row.get('actual_delivery_date')) or date.today())
                        if st.form_submit_button("Update", type="primary"):
                            conn = sqlite3.connect(DB_FILE)
                            conn.execute("UPDATE tasks_v2 SET status=?, actual_delivery_date=? WHERE id=?", (ns, str(ad), row['id']))
                            conn.commit(); conn.close()
                            st.success("Updated!"); st.rerun()

# --- TRAINING APP ---
def app_training():
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("⬅ Home", use_container_width=True): st.session_state['current_app']='HOME'; st.rerun()
    with c2: st.markdown("### 🎓 Training Hub")
    st.markdown("---")

    if st.session_state['role'] == "Team Leader":
        t1, t2 = st.tabs(["Repository", "Add New"])
        with t1:
            df = get_trainings()
            with st.expander("📂 Import / Export"):
                col_imp, col_exp = st.columns(2)
                with col_imp:
                    up_train = st.file_uploader("Upload CSV", type=['csv'])
                    if up_train:
                        if import_training_csv(up_train): st.rerun()
                with col_exp:
                    if not df.empty:
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download CSV", data=csv, file_name="training_repo.csv", mime="text/csv")
            if not df.empty: st.dataframe(df, use_container_width=True)
            else: st.info("Repository empty.")
        with t2:
            with st.form("add_training_form"):
                tt = st.text_input("Title")
                td = st.text_area("Desc")
                tl = st.text_input("Link")
                tm = st.checkbox("Mandatory")
                if st.form_submit_button("Publish", type="primary", use_container_width=True):
                    add_training(tt, td, tl, "All", tm, st.session_state['name'])
                    st.success("Published."); st.rerun()
    else:
        df = get_trainings(user_name=st.session_state['name'])
        if not df.empty:
            comp = len(df[df['status']=='Completed'])
            st.progress(comp/len(df), text=f"Progress: {int((comp/len(df))*100)}%")
        st.markdown("#### Modules")
        if df.empty: st.info("No training found.")
        else:
            for idx, row in df.iterrows():
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**{row['title']}**")
                        st.caption(row['description'])
                        st.markdown(f"[{row['link']}]({row['link']})")
                    with c2:
                        c_stat = row['status']
                        n_stat = st.selectbox("Status", ["Not Started", "In Progress", "Completed"], 
                                              index=["Not Started", "In Progress", "Completed"].index(c_stat), 
                                              key=f"tr_stat_{row['id']}", label_visibility="collapsed")
                        if n_stat != c_stat:
                            update_training_status(st.session_state['name'], row['id'], n_stat); st.rerun()

# --- RESOURCE TRACKER (Formerly Onboarding) ---
def app_resource():
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("⬅ Home", use_container_width=True): st.session_state['current_app']='HOME'; st.rerun()
    with c2: st.markdown("### 🚀 Resource Tracker")
    st.markdown("---")

    # --- TEAM LEADER VIEW: CARD LINE ITEMS & EDIT ---
    if st.session_state['role'] == "Team Leader":
        # State for editing
        if 'edit_res_user' not in st.session_state: st.session_state['edit_res_user'] = None
        if 'add_res_mode' not in st.session_state: st.session_state['add_res_mode'] = False

        # Header Row: Title + Add Button
        h1, h2 = st.columns([4, 1])
        with h1: st.markdown("#### Team Resources")
        with h2: 
            if st.button("➕ Add Resource", type="primary", use_container_width=True):
                st.session_state['add_res_mode'] = True
                st.session_state['edit_res_user'] = None
                st.rerun()

        # ADD / EDIT FORM
        if st.session_state['add_res_mode'] or st.session_state['edit_res_user']:
            with st.container(border=True):
                is_add = st.session_state['add_res_mode']
                st.subheader("Add Resource" if is_add else f"Edit Resource: {st.session_state['edit_res_user']}")
                
                # Fetch existing data if editing
                def_data = {}
                if not is_add:
                    df_res = get_all_onboarding()
                    row = df_res[df_res['username'] == st.session_state['edit_res_user']]
                    if not row.empty: def_data = row.iloc[0].to_dict()

                with st.form("resource_form"):
                    c1, c2, c3 = st.columns(3)
                    # For Add Mode, we need manual inputs. For Edit, they might be read-only if from Login
                    uname = c1.text_input("Username (Unique)", value=def_data.get('username',''), disabled=not is_add)
                    fname = c2.text_input("Full Name", value=def_data.get('fullname',''))
                    empid = c3.text_input("Emp ID", value=def_data.get('emp_id',''))
                    
                    c4, c5 = st.columns(2)
                    tid_val = c4.text_input("TID", value=def_data.get('tid',''))
                    loc_val = c5.selectbox("Location", ["Chennai", "Pune", "Bangalore", "Client Site"], index=0)
                    
                    st.markdown("---")
                    st.markdown("**Access Control (Manager)**")
                    m1, m2, m3 = st.columns(3)
                    t_act = m1.checkbox("TID Active", value=bool(def_data.get('tid_active', 0)))
                    e_mail = m2.checkbox("Ext. Mail Created", value=bool(def_data.get('ext_mail_id', 0)))
                    tc_acc = m3.checkbox("Teamcenter Access", value=bool(def_data.get('teamcenter_access', 0)))

                    if st.form_submit_button("💾 Save Resource"):
                        if not uname: st.error("Username required"); st.stop()
                        payload = {
                            "username": uname, "fullname": fname, "emp_id": empid, "tid": tid_val,
                            "location": loc_val, "tid_active": 1 if t_act else 0,
                            "ext_mail_id": 1 if e_mail else 0, "teamcenter_access": 1 if tc_acc else 0,
                            # Preserve or default others
                            "work_mode": def_data.get('work_mode', 'Office'),
                            "hr_policy_briefing": def_data.get('hr_policy_briefing', 0),
                            "it_system_setup": def_data.get('it_system_setup', 0),
                            "team_centre_training": def_data.get('team_centre_training', 0),
                            "agt_access": def_data.get('agt_access', 0),
                            "rdp_access": def_data.get('rdp_access', 0),
                            "avd_access": def_data.get('avd_access', 0),
                            "blocking_point": def_data.get('blocking_point', ''),
                            "ticket_raised": def_data.get('ticket_raised', '')
                        }
                        save_onboarding_details(payload)
                        st.success("Resource Saved")
                        st.session_state['add_res_mode'] = False
                        st.session_state['edit_res_user'] = None
                        st.rerun()
                
                if st.button("Cancel"):
                    st.session_state['add_res_mode'] = False
                    st.session_state['edit_res_user'] = None
                    st.rerun()
            st.markdown("---")

        # DISPLAY CARDS
        df = get_all_onboarding()
        if not df.empty:
            for idx, row in df.iterrows():
                with st.container(border=True):
                    # Layout: Avatar | Details | Status | Action
                    c1, c2, c3, c4 = st.columns([1, 3, 2, 1])
                    with c1:
                        st.markdown("👤") # Placeholder icon
                    with c2:
                        st.markdown(f"**{row['fullname']}**")
                        st.caption(f"{row['emp_id']} | {row['tid']}")
                    with c3:
                        st.caption("Onboarding Status")
                        # Simple visual check
                        if row['tid_active'] and row['ext_mail_id']:
                            st.markdown("<span class='status-ok'>Active</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("<span class='status-pending'>Pending Setup</span>", unsafe_allow_html=True)
                    with c4:
                        if st.button("Edit", key=f"res_edit_{row['username']}"):
                            st.session_state['edit_res_user'] = row['username']
                            st.session_state['add_res_mode'] = False
                            st.rerun()
        else:
            st.info("No resources found. Click 'Add Resource' to begin.")

    # --- MEMBER VIEW (Checklist) ---
    else:
        df = get_onboarding_details(st.session_state['user'])
        defaults = df.iloc[0].to_dict() if not df.empty else {}
        with st.container(border=True):
            st.subheader("Resource Checklist")
            st.markdown("##### 👤 Employee Details")
            ac1, ac2, ac3 = st.columns(3)
            ac1.text_input("Full Name", value=st.session_state['name'], disabled=True)
            ac2.text_input("Employee ID", value=st.session_state['emp_id'], disabled=True)
            ac3.text_input("TID", value=st.session_state['tid'], disabled=True)

            with st.form("onboarding_form"):
                lc1, lc2 = st.columns(2)
                loc = lc1.selectbox("Location", ["Chennai", "Pune", "Bangalore", "Client Site"], 
                                    index=["Chennai", "Pune", "Bangalore", "Client Site"].index(defaults.get('location', 'Chennai')))
                work_mode = lc2.selectbox("Work Mode", ["Office", "Remote"], 
                                          index=["Office", "Remote"].index(defaults.get('work_mode', 'Office')))

                st.markdown("---")
                st.markdown("##### ✅ Checklist")
                r1c1, r1c2, r1c3 = st.columns(3)
                hr_pol = r1c1.checkbox("HR Policy Briefing", value=bool(defaults.get('hr_policy_briefing', 0)))
                it_set = r1c2.checkbox("IT System Setup", value=bool(defaults.get('it_system_setup', 0)))
                tc_trn = r1c3.checkbox("Team Centre Training", value=bool(defaults.get('team_centre_training', 0)))

                st.write("")
                st.markdown("**Remote / Access Validations**")
                r2c1, r2c2, r2c3 = st.columns(3)
                agt_val = bool(defaults.get('agt_access', 0))
                rdp_val = bool(defaults.get('rdp_access', 0))
                avd_val = bool(defaults.get('avd_access', 0))
                
                if work_mode == "Remote":
                    r2c1.markdown("*(Required)*")
                    agt = r2c1.checkbox("AGT Access", value=agt_val)
                    r2c2.markdown("*(Required)*")
                    rdp = r2c2.checkbox("RDP Access", value=rdp_val)
                    r2c3.markdown("*(Required)*")
                    avd = r2c3.checkbox("AVD Access", value=avd_val)
                else:
                    r2c1.markdown("*(N/A)*"); agt = r2c1.checkbox("AGT Access", value=agt_val, disabled=True)
                    r2c2.markdown("*(N/A)*"); rdp = r2c2.checkbox("RDP Access", value=rdp_val, disabled=True)
                    r2c3.markdown("*(N/A)*"); avd = r2c3.checkbox("AVD Access", value=avd_val, disabled=True)

                st.markdown("---")
                st.markdown("##### 🔒 Manager / IT Approvals (Read Only)")
                m1, m2, m3 = st.columns(3)
                m1.checkbox("TID Active", value=bool(defaults.get('tid_active', 0)), disabled=True)
                m2.checkbox("External Mail ID", value=bool(defaults.get('ext_mail_id', 0)), disabled=True)
                m3.checkbox("Teamcenter Access", value=bool(defaults.get('teamcenter_access', 0)), disabled=True)

                st.markdown("---")
                st.markdown("##### ⚠️ Issues")
                bp = st.text_input("Blocking Point (If any)", value=defaults.get('blocking_point', ''))
                
                ticket_action = defaults.get('ticket_raised', '')
                raise_ticket = False
                if bp:
                    st.warning("Blocking point detected.")
                    if ticket_action: st.success(f"Ticket Raised: {ticket_action}")
                    else: raise_ticket = st.checkbox("Raise IT Ticket?")

                if st.form_submit_button("💾 Save Form", type="primary"):
                    new_ticket_status = ticket_action
                    if raise_ticket and not ticket_action: new_ticket_status = f"TKT-{str(uuid.uuid4())[:6]}"
                    payload = {
                        "username": st.session_state['user'], "fullname": st.session_state['name'],
                        "emp_id": st.session_state['emp_id'], "tid": st.session_state['tid'],
                        "location": loc, "work_mode": work_mode,
                        "hr_policy_briefing": 1 if hr_pol else 0, "it_system_setup": 1 if it_set else 0,
                        "tid_active": defaults.get('tid_active', 0), "team_centre_training": 1 if tc_trn else 0,
                        "agt_access": 1 if agt else 0, "ext_mail_id": defaults.get('ext_mail_id', 0),
                        "rdp_access": 1 if rdp else 0, "avd_access": 1 if avd else 0,
                        "teamcenter_access": defaults.get('teamcenter_access', 0),
                        "blocking_point": bp, "ticket_raised": new_ticket_status
                    }
                    save_onboarding_details(payload); st.success("Saved!"); st.rerun()

# ---------- MAIN CONTROLLER ----------
def main():
    init_db()
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['current_app'] = 'HOME'

    if st.session_state['logged_in']:
        with st.sidebar:
            img_url = st.session_state.get('img', '')
            if img_url: st.markdown(f"<img src='{img_url}' class='profile-img'>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align:center;'>{st.session_state.get('name','')}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; color:gray;'>{st.session_state.get('role','')}</p>", unsafe_allow_html=True)
            st.markdown("---")
            if st.button("Sign Out", use_container_width=True): st.session_state.clear(); st.rerun()

    if not st.session_state['logged_in']:
        login_page()
    else:
        app = st.session_state.get('current_app', 'HOME')
        if app == 'HOME': app_home()
        elif app == 'KPI': app_kpi()
        elif app == 'TRAINING': app_training()
        elif app == 'RESOURCE': app_resource() # Renamed Routing

if __name__ == "__main__":
    main()
