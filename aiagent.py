import streamlit as st
import pandas as pd
import sqlite3
import uuid
from datetime import date, datetime, timedelta
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
    .status-active { color: #10b981; font-weight: bold; }
    .status-inactive { color: #ef4444; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- DATABASE ----------
DB_FILE = "portal_data_final_v15_full.db"

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
    
    # 3. RESOURCE TRACKER V2 (Updated Schema)
    c.execute('''CREATE TABLE IF NOT EXISTS resource_tracker_v2 (
        employee_id TEXT PRIMARY KEY,
        employee_name TEXT,
        dev_role TEXT,
        department TEXT,
        location TEXT,
        reporting_manager TEXT,
        onboarding_start_date TEXT,
        skill_level TEXT,
        system_access_req TEXT,
        mandatory_trainings TEXT,
        doc_list_req TEXT,
        po_details TEXT,
        status TEXT,
        remarks TEXT,
        effective_exit_date TEXT,
        backfill_status TEXT,
        reason_for_leaving TEXT
    )''')
    
    # --- DEMO DATA INJECTION ---
    c.execute("SELECT count(*) FROM training_repo")
    if c.fetchone()[0] == 0:
        trainings = [
            ("TR-01", "Python Basics", "Introduction to Python syntax", "https://python.org", "All", 1, "System"),
            ("TR-02", "Advanced Pandas", "Data manipulation mastery", "https://pandas.pydata.org", "Team Member", 0, "System"),
            ("TR-03", "Streamlit UI", "Building interactive dashboards", "https://streamlit.io", "All", 1, "System"),
        ]
        c.executemany("INSERT INTO training_repo VALUES (?,?,?,?,?,?,?)", trainings)

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

# --- RESOURCE TRACKER V2 HELPERS ---
def get_all_resources_v2():
    conn = sqlite3.connect(DB_FILE)
    try: 
        df = pd.read_sql_query("SELECT * FROM resource_tracker_v2", conn)
    except: 
        df = pd.DataFrame()
    conn.close()
    return df

def save_resource_v2(data):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Check if Employee ID exists
    c.execute("SELECT employee_id FROM resource_tracker_v2 WHERE employee_id=?", (data['employee_id'],))
    exists = c.fetchone()
    
    cols = [
        'employee_id', 'employee_name', 'dev_role', 'department', 'location', 
        'reporting_manager', 'onboarding_start_date', 'skill_level', 'system_access_req',
        'mandatory_trainings', 'doc_list_req', 'po_details', 'status', 'remarks',
        'effective_exit_date', 'backfill_status', 'reason_for_leaving'
    ]
    
    vals = [data.get(k) for k in cols]
    
    if exists:
        set_clause = ", ".join([f"{col}=?" for col in cols])
        # Append ID to end of vals for the WHERE clause (assuming ID is the first col in vals but used as key)
        vals.append(data['employee_id']) 
        c.execute(f"UPDATE resource_tracker_v2 SET {set_clause} WHERE employee_id=?", vals)
    else:
        placeholders = ",".join(["?"] * len(cols))
        c.execute(f"INSERT INTO resource_tracker_v2 VALUES ({placeholders})", vals)
        
    conn.commit()
    conn.close()

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
    st.caption(f"ID: {st.session_state.get('emp_id')} | Role: {st.session_state.get('role')}")
    st.write("---")
    
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
            st.markdown("### 🚀 **Resource Tracker**"); st.caption("Staffing & Exit Mgmt")
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

# --- RESOURCE TRACKER APP (UPDATED V2) ---
def app_resource():
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("⬅ Home", use_container_width=True): st.session_state['current_app']='HOME'; st.rerun()
    with c2: st.markdown("### 🚀 Resource Tracker")
    st.markdown("---")

    # Only Team Leaders can Manage Resources
    if st.session_state['role'] != "Team Leader":
        st.warning("Access Restricted: Only Team Leaders can manage the Resource Tracker.")
        return

    # Initialization
    if 'edit_res_id' not in st.session_state: st.session_state['edit_res_id'] = None
    if 'add_res_mode' not in st.session_state: st.session_state['add_res_mode'] = False

    # --- Header & Actions ---
    h1, h2 = st.columns([4, 1])
    with h1: st.markdown("#### Team Staffing Details")
    with h2: 
        if st.button("➕ Add Resource", type="primary", use_container_width=True):
            st.session_state['add_res_mode'] = True
            st.session_state['edit_res_id'] = None
            st.rerun()

    # --- ADD / EDIT FORM ---
    if st.session_state['add_res_mode'] or st.session_state['edit_res_id']:
        with st.container(border=True):
            is_add = st.session_state['add_res_mode']
            st.subheader("Resource Details" if is_add else f"Edit Resource: {st.session_state['edit_res_id']}")
            
            # Load Data if Editing
            d = {}
            if not is_add:
                df_all = get_all_resources_v2()
                row = df_all[df_all['employee_id'] == st.session_state['edit_res_id']]
                if not row.empty: d = row.iloc[0].to_dict()

            with st.form("resource_v2_form"):
                # ROW 1: Basic Info
                c1, c2, c3 = st.columns(3)
                emp_name = c1.text_input("Employee Name", value=d.get('employee_name', ''))
                emp_id = c2.text_input("Employee ID", value=d.get('employee_id', ''), disabled=not is_add)
                
                # Auto-fill Manager
                manager_val = d.get('reporting_manager', st.session_state['name'])
                rep_manager = c3.selectbox("Reporting Manager", [st.session_state['name'], "Jane Doe", "John Smith"], 
                                           index=0 if manager_val == st.session_state['name'] else 0)

                # ROW 2: Role & Location
                c4, c5, c6 = st.columns(3)
                dev_role = c4.selectbox("DEV / Role", ["Frontend Dev", "Backend Dev", "Fullstack", "QA", "Data Scientist"], 
                                        index=["Frontend Dev", "Backend Dev", "Fullstack", "QA", "Data Scientist"].index(d.get('dev_role', 'Frontend Dev')) if d.get('dev_role') in ["Frontend Dev", "Backend Dev", "Fullstack", "QA", "Data Scientist"] else 0)
                
                dept = c5.selectbox("Department", ["Engineering", "Product", "Design", "Data"], 
                                    index=["Engineering", "Product", "Design", "Data"].index(d.get('department', 'Engineering')) if d.get('department') in ["Engineering", "Product", "Design", "Data"] else 0)
                
                loc_list = ["Chennai", "Pune", "Bangalore", "Hyderabad", "Remote"]
                loc = c6.selectbox("Location", loc_list, index=loc_list.index(d.get('location', 'Chennai')) if d.get('location') in loc_list else 0)

                # ROW 3: Dates & Skills
                c7, c8 = st.columns(2)
                
                # Date Parse Helper
                def get_date_val(key):
                    if d.get(key) and d.get(key) != 'None':
                        try: return pd.to_datetime(d.get(key)).date()
                        except: return date.today()
                    return date.today()

                onboard_date = c7.date_input("Onboarding Start Date", value=get_date_val('onboarding_start_date'))
                skill = c8.selectbox("Skill Level", ["L1 - Junior", "L2 - Intermediate", "L3 - Senior", "L4 - Lead"], 
                                     index=["L1 - Junior", "L2 - Intermediate", "L3 - Senior", "L4 - Lead"].index(d.get('skill_level', 'L1 - Junior')) if d.get('skill_level') else 0)

                # ROW 4: Systems & Auto-Derived
                st.markdown("---")
                st.caption("🔒 System & Compliance")
                
                # System Access (Multi-select)
                saved_sys = d.get('system_access_req', '').split(',') if d.get('system_access_req') else []
                sys_req = st.multiselect("System Access Requirements", ["JIRA", "Confluence", "GitLab", "AWS", "Azure", "SAP"], default=saved_sys)
                
                # Auto-Derived Logic
                derived_training = "Security Awareness, Code of Conduct"
                if dev_role == "QA": derived_training += ", Automation Basics"
                if dept == "Data": derived_training += ", GDPR Training"
                
                derived_docs = "PAN Card, Aadhar"
                if loc != "Remote": derived_docs += ", Vaccine Certificate"

                cc1, cc2 = st.columns(2)
                cc1.text_area("Mandatory Role-based Trainings (Auto)", value=derived_training, disabled=True)
                cc2.text_area("Document List Required (Auto)", value=derived_docs, disabled=True)
                
                po_det = st.text_input("PO Details", value=d.get('po_details', ''))

                # ROW 5: Status & Conditional Logic
                st.markdown("---")
                st.caption("⚡ Status & Exit Management")
                
                stat_opts = ["Active", "Inactive"]
                curr_stat = d.get('status', 'Active')
                status = st.selectbox("Status", stat_opts, index=stat_opts.index(curr_stat) if curr_stat in stat_opts else 0)
                
                rem = st.text_area("Remarks if any", value=d.get('remarks', ''))

                # CONDITIONAL INPUTS
                exit_date = None
                backfill = "N/A"
                reason = ""
                
                if status == "Inactive":
                    st.error("User marked as Inactive. Please fill exit details below.")
                    ec1, ec2 = st.columns(2)
                    exit_date = ec1.date_input("Effective Exit Day", value=get_date_val('effective_exit_date'))
                    backfill = ec2.selectbox("Backfill Status", ["Initiated", "Pending Approval", "Interviews On-going", "Closed", "Not Required"], 
                                             index=0 if not d.get('backfill_status') else ["Initiated", "Pending Approval", "Interviews On-going", "Closed", "Not Required"].index(d.get('backfill_status')))
                    reason = st.text_area("Reason for Leaving from Project", value=d.get('reason_for_leaving', ''))

                # SAVE BUTTON
                if st.form_submit_button("💾 Save Resource Details", type="primary", use_container_width=True):
                    if not emp_id or not emp_name:
                        st.error("Employee Name and ID are required!")
                        st.stop()
                    
                    payload = {
                        "employee_id": emp_id, "employee_name": emp_name, "dev_role": dev_role,
                        "department": dept, "location": loc, "reporting_manager": rep_manager,
                        "onboarding_start_date": str(onboard_date), "skill_level": skill,
                        "system_access_req": ",".join(sys_req), # Store list as CSV string
                        "mandatory_trainings": derived_training,
                        "doc_list_req": derived_docs,
                        "po_details": po_det, "status": status, "remarks": rem,
                        "effective_exit_date": str(exit_date) if exit_date else None,
                        "backfill_status": backfill if status == "Inactive" else None,
                        "reason_for_leaving": reason if status == "Inactive" else None
                    }
                    save_resource_v2(payload)
                    st.success("Resource Saved Successfully!")
                    st.session_state['add_res_mode'] = False
                    st.session_state['edit_res_id'] = None
                    st.rerun()

            if st.button("Cancel"):
                st.session_state['add_res_mode'] = False
                st.session_state['edit_res_id'] = None
                st.rerun()

    # --- LIST VIEW ---
    st.markdown("### 📋 Staff Directory")
    df = get_all_resources_v2()
    
    if not df.empty:
        # Simple Filter
        f_stat = st.selectbox("Filter by Status", ["All", "Active", "Inactive"], index=1)
        if f_stat != "All":
            df = df[df['status'] == f_stat]

        for idx, row in df.iterrows():
            with st.container(border=True):
                c_main, c_det, c_stat, c_act = st.columns([3, 3, 2, 1])
                
                with c_main:
                    st.markdown(f"**{row['employee_name']}**")
                    st.caption(f"ID: {row['employee_id']}")
                    st.caption(f"🛠 {row['dev_role']}")

                with c_det:
                    st.write(f"📍 {row['location']}")
                    st.write(f"📅 Onboard: {row['onboarding_start_date']}")
                    if row['status'] == "Inactive":
                        st.write(f"🚪 Exit: {row['effective_exit_date']}")

                with c_stat:
                    color = "#10b981" if row['status'] == 'Active' else "#ef4444"
                    st.markdown(f"<span style='color:{color}; font-weight:bold; font-size:1.1em;'>● {row['status']}</span>", unsafe_allow_html=True)
                    if row['status'] == "Inactive":
                        st.caption(f"Reason: {row['reason_for_leaving']}")

                with c_act:
                    if st.button("✏️", key=f"edit_v2_{row['employee_id']}"):
                        st.session_state['edit_res_id'] = row['employee_id']
                        st.session_state['add_res_mode'] = False
                        st.rerun()
    else:
        st.info("No resources found in the new tracker. Click 'Add Resource' to get started.")

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
        elif app == 'RESOURCE': app_resource()

if __name__ == "__main__":
    main()
