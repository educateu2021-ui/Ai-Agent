import streamlit as st
import pandas as pd
import sqlite3
import uuid
from datetime import date, datetime, timedelta
import random
import plotly.express as px
import plotly.graph_objects as go
import string

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
    .status-warn { color: #f59e0b; font-weight: bold; }
    .status-bad { color: #ef4444; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- DATABASE & SEEDING ----------
DB_FILE = "portal_v22_ultimate.db"

def seed_data(c):
    """
    Uses INSERT OR REPLACE to FORCE specific accounts to exist with specific passwords.
    """
    
    # 1. FORCE CREATE FIXED USERS (This overwrites them if they exist to fix passwords)
    # (Username, Password, Role, Name, EmpID)
    mandatory_users = [
        ("admin", "admin123", "Super Admin", "System Admin", "ADM-000"),
        ("leader", "123", "Team Leader", "Sarah Jenkins", "LDR-001"),
        ("member", "123", "Team Member", "David Chen", "EMP-101")
    ]

    for u_user, u_pass, u_role, u_name, u_id in mandatory_users:
        img = f"https://ui-avatars.com/api/?name={u_name.replace(' ','+')}&background=random"
        # REPLACE ensures password is reset to what we expect
        c.execute("INSERT OR REPLACE INTO users (username, password, role, name, emp_id, img, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                  (u_user, u_pass, u_role, u_name, u_id, img, str(date.today())))

    # 2. FILL RANDOM KPI TASKS (Only if table empty)
    c.execute("SELECT count(*) FROM tasks_v2")
    if c.fetchone()[0] == 0:
        c.execute("SELECT name FROM users WHERE role='Team Member'")
        pilots = [row[0] for row in c.fetchall()]
        if not pilots: pilots = ["David Chen"]

        for i in range(1, 21):
            pilot = random.choice(pilots)
            status = random.choice(["Completed", "Inprogress", "Hold", "Cancelled"])
            start = date.today() - timedelta(days=random.randint(10, 60))
            due = start + timedelta(days=random.randint(5, 20))
            
            actual, otd = "", "N/A"
            if status == "Completed":
                delay = random.choice([-2, -1, 0, 1, 5])
                actual_dt = due + timedelta(days=delay)
                actual = str(actual_dt)
                otd = "OK" if actual_dt <= due else "NOT OK"

            c.execute("INSERT INTO tasks_v2 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (str(uuid.uuid4())[:8], pilot, f"Project Task {i:02d}", str(start), actual, str(due),
                       status, "Yes", f"REF-{1000+i}", "Yes", otd, 
                       f"Description for task {i}", "Standard", 
                       "Yes", str(start), str(start), otd, "None", "QA-Ref", "Lead-X", "Mgr-Y"))

    # 3. FILL TRAINING (Only if empty)
    c.execute("SELECT count(*) FROM training_repo")
    if c.fetchone()[0] == 0:
        topics = ["Python Basics", "Safety Protocols", "Leadership 101", "Agile", "Communication", "Data Privacy", "Cyber Security", "Excel Advanced", "Power BI", "SQL Funda"]
        for t in topics:
            c.execute("INSERT INTO training_repo VALUES (?,?,?,?,?,?,?)",
                      (str(uuid.uuid4())[:8], t, f"Learn about {t}", "http://example.com", 
                       random.choice(["All", "Team Leader", "Team Member"]), random.choice([0, 1]), "System"))

    # 4. FILL RESOURCES (Only if empty)
    c.execute("SELECT count(*) FROM resource_tracker_v4")
    if c.fetchone()[0] == 0:
        depts = ["Engineering", "Quality", "Manufacturing"]
        locs = ["Chennai", "Bangalore", "Pune"]
        for i in range(10):
            status = random.choice(["Active", "Active", "Inactive"])
            exit_date = str(date.today()) if status == "Inactive" else ""
            reason = "Resigned" if status == "Inactive" else ""
            c.execute("INSERT INTO resource_tracker_v4 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", 
                      (str(uuid.uuid4())[:8], f"Resource {i}", f"RES-{i}", "001", 
                       random.choice(depts), random.choice(locs), "Sarah Jenkins", str(date.today()),
                       "MID", status, "PO-123", "", exit_date, "No", reason, 
                       str(random.randint(20, 50)), "5"))

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Create Tables
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY, password TEXT, role TEXT, name TEXT, 
        emp_id TEXT, img TEXT, created_at TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS tasks_v2 (
        id TEXT PRIMARY KEY, name_activity_pilot TEXT, task_name TEXT, date_of_receipt TEXT,
        actual_delivery_date TEXT, commitment_date_to_customer TEXT, status TEXT,
        ftr_customer TEXT, reference_part_number TEXT, ftr_internal TEXT, otd_internal TEXT,
        description_of_activity TEXT, activity_type TEXT, ftr_quality_gate_internal TEXT,
        date_of_clarity_in_input TEXT, start_date TEXT, otd_customer TEXT, customer_remarks TEXT,
        name_quality_gate_referent TEXT, project_lead TEXT, customer_manager_name TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS training_repo (
        id TEXT PRIMARY KEY, title TEXT, description TEXT, link TEXT, 
        role_target TEXT, mandatory INTEGER, created_by TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS training_progress (
        user_name TEXT, training_id TEXT, status TEXT, 
        last_updated TEXT, PRIMARY KEY (user_name, training_id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS resource_tracker_v4 (
        id TEXT PRIMARY KEY, employee_name TEXT, employee_id TEXT, dev_code TEXT,
        department TEXT, location TEXT, reporting_manager TEXT, onboarding_date TEXT,
        experience_level TEXT, status TEXT, po_details TEXT, remarks TEXT,
        effective_exit_date TEXT, backfill_status TEXT, reason_for_leaving TEXT,
        hourly_rate TEXT, hardware_daily_cost TEXT)''')
    
    seed_data(c)
    conn.commit()
    conn.close()

# ---------- UTILS & HELPERS ----------

def generate_temp_password(length=8):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for i in range(length))

def get_all_users():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM users", conn)
    conn.close()
    return df

def save_user_entry(data, is_update=False):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    if is_update:
        c.execute("UPDATE users SET password=?, role=?, name=?, emp_id=?, img=? WHERE username=?",
                  (data['password'], data['role'], data['name'], data['emp_id'], data['img'], data['username']))
    else:
        c.execute("INSERT OR REPLACE INTO users VALUES (?,?,?,?,?,?,?)",
                  (data['username'], data['password'], data['role'], data['name'], data['emp_id'], data['img'], str(date.today())))
    conn.commit()
    conn.close()

def delete_user(username):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    conn.close()

def import_users_csv(file):
    try:
        df = pd.read_csv(file)
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        for _, row in df.iterrows():
            c.execute("INSERT OR REPLACE INTO users VALUES (?,?,?,?,?,?,?)",
                      (row['username'], row['password'], row['role'], row['name'], 
                       row.get('emp_id',''), row.get('img',''), str(date.today())))
        conn.commit()
        conn.close()
        return True
    except: return False

# --- NEW HELPERS FOR PROFILE ---
def get_user_resource_details(emp_id):
    """Fetches details from resource_tracker based on Employee ID (excluding costs)"""
    conn = sqlite3.connect(DB_FILE)
    try:
        # Fetching all, filtering in pandas for simplicity to handle loose matches
        df = pd.read_sql_query("SELECT * FROM resource_tracker_v4 WHERE employee_id=?", conn, params=(emp_id,))
    except: 
        df = pd.DataFrame()
    conn.close()
    return df

def update_user_credentials(username, new_password=None, new_img=None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    if new_password:
        c.execute("UPDATE users SET password=? WHERE username=?", (new_password, username))
    if new_img:
        c.execute("UPDATE users SET img=? WHERE username=?", (new_img, username))
    conn.commit()
    conn.close()

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
        conn = sqlite3.connect(DB_FILE)
        df.to_sql('tasks_v2', conn, if_exists='append', index=False)
        conn.close()
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

def delete_training(tid):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM training_repo WHERE id=?", (tid,))
    conn.commit()
    conn.close()

def delete_all_trainings():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM training_repo")
    c.execute("DELETE FROM training_progress")
    conn.commit()
    conn.close()

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
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        for _, row in df.iterrows():
            tid = str(uuid.uuid4())[:8]
            c.execute("INSERT INTO training_repo VALUES (?,?,?,?,?,?,?)",
                      (tid, row.get('title','No Title'), row.get('description',''), 
                       row.get('link','#'), row.get('role_target','All'), 
                       int(row.get('mandatory', 0)), 'Imported'))
        conn.commit()
        conn.close()
        return True
    except: return False

# --- RESOURCE TRACKER HELPERS ---
def get_resource_list():
    conn = sqlite3.connect(DB_FILE)
    try: df = pd.read_sql_query("SELECT * FROM resource_tracker_v4", conn)
    except: df = pd.DataFrame()
    conn.close()
    return df

def save_resource_entry(data, res_id=None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    cols = ['employee_name', 'employee_id', 'dev_code', 'department', 'location', 
            'reporting_manager', 'onboarding_date', 'experience_level', 'status', 
            'po_details', 'remarks', 'effective_exit_date', 'backfill_status', 
            'reason_for_leaving', 'hourly_rate', 'hardware_daily_cost']
    vals = [str(data.get(k, '')) for k in cols]
    
    if res_id:
        # Update existing
        set_clause = ", ".join([f"{col}=?" for col in cols])
        c.execute(f"UPDATE resource_tracker_v4 SET {set_clause} WHERE id=?", (*vals, res_id))
        conn.commit(); conn.close()
        return None
    else:
        # Create new
        new_id = str(uuid.uuid4())[:8]
        placeholders = ",".join(["?"] * (len(cols) + 1))
        c.execute(f"INSERT INTO resource_tracker_v4 VALUES ({placeholders})", (new_id, *vals))
        
        # --- AUTO CREATE USER LOGIN ---
        # Logic: username = empid_lowercase, password = auto-generated
        emp_id = data.get('employee_id', 'unknown')
        username = emp_id.lower().replace(" ", "")
        temp_pass = generate_temp_password()
        name = data.get('employee_name', 'New User')
        role = "Team Member" 
        img = f"https://ui-avatars.com/api/?name={name.replace(' ','+')}&background=random"
        
        # Insert user only if username doesn't exist
        c.execute("SELECT count(*) FROM users WHERE username=?", (username,))
        if c.fetchone()[0] == 0:
            c.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?)", 
                      (username, temp_pass, role, name, emp_id, img, str(date.today())))
            conn.commit(); conn.close()
            return f"User: {username} | Pass: {temp_pass}"
        
        conn.commit(); conn.close()
        return None

def import_resource_csv(file):
    try:
        df = pd.read_csv(file)
        cols = ['employee_name', 'employee_id', 'dev_code', 'department', 'location', 
                'reporting_manager', 'onboarding_date', 'experience_level', 'status', 
                'po_details', 'remarks', 'effective_exit_date', 'backfill_status', 
                'reason_for_leaving', 'hourly_rate', 'hardware_daily_cost']
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        for _, row in df.iterrows():
            new_id = str(uuid.uuid4())[:8]
            vals = [str(row.get(k, '')) for k in cols]
            placeholders = ",".join(["?"] * (len(cols) + 1))
            c.execute(f"INSERT INTO resource_tracker_v4 VALUES ({placeholders})", (new_id, *vals))
        conn.commit()
        conn.close()
        return True
    except: return False

# --- PLOTLY HELPERS ---
def get_analytics_chart(df):
    if df.empty: return go.Figure()
    df_local = df.copy()
    status_counts = df_local['status'].value_counts()
    fig = px.bar(x=status_counts.index, y=status_counts.values, color=status_counts.index,
                 color_discrete_map={"Completed":"#10b981","Inprogress":"#3b82f6","Hold":"#f59e0b","Cancelled":"#ef4444"})
    fig.update_layout(xaxis_title="Status", yaxis_title="Count", height=300, showlegend=False, margin=dict(l=0,r=0,t=10,b=0))
    return fig

def get_donut(df):
    if df.empty: return go.Figure()
    ftr_yes = len(df[df['ftr_internal']=='Yes'])
    total = len(df)
    pct = int((ftr_yes/total)*100) if total>0 else 0
    fig = go.Figure(data=[go.Pie(labels=['FTR OK','FTR NOT OK'], values=[ftr_yes, total-ftr_yes], hole=.7, textinfo='none', marker_colors=['#10b981', '#ef4444'])])
    fig.update_layout(height=240, margin=dict(l=0,r=0,t=0,b=0), 
                      annotations=[dict(text=f"FTR {pct}%", x=0.5, y=0.5, showarrow=False, font=dict(size=16))])
    return fig

# ---------- AUTH ----------
def login_page():
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        with st.container(border=True):
            st.markdown("<h2 style='text-align:center; color:#1f2937;'>Portal Sign In</h2>", unsafe_allow_html=True)
            
            # Use lower() on input to normalize
            u = st.text_input("Username").strip() 
            p = st.text_input("Password", type="password").strip()
            
            if st.button("Secure Login", use_container_width=True, type="primary"):
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                
                # UPDATED QUERY: Checks lowercase username matches lowercase input
                c.execute("SELECT * FROM users WHERE LOWER(username)=? AND password=?", (u.lower(), p))
                user_data = c.fetchone()
                conn.close()
                
                if user_data:
                    st.session_state.update({
                        'logged_in': True, 'user': user_data[0], 'role': user_data[2], 
                        'name': user_data[3], 'emp_id': user_data[4],
                        'img': user_data[5], 'current_app': 'HOME'
                    })
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
                    
            # DEBUGGING HELPER (Optional: Remove before deployment)
            with st.expander("Debug: View Valid Users"):
                conn = sqlite3.connect(DB_FILE)
                debug_df = pd.read_sql("SELECT username, role, password FROM users", conn)
                st.dataframe(debug_df)
                conn.close()
# ---------- APP SECTIONS ----------
def app_home():
    st.markdown(f"## Welcome, {st.session_state['name']}")
    st.caption(f"ID: {st.session_state.get('emp_id')} | Role: {st.session_state.get('role')}")
    st.write("---")
    
    if st.session_state['role'] == 'Super Admin':
        c_adm, c1, c2, c3 = st.columns(4)
        with c_adm:
            with st.container(border=True):
                st.markdown("### 🛡️ **User Admin**")
                st.caption("Manage Users & Access")
                if st.button("Manage Users", use_container_width=True, type="primary"): 
                    st.session_state['current_app']='ADMIN'
                    st.rerun()
    else:
        c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.markdown("### 📊 **KPI System**"); st.caption("Manage OTD & FTR")
            if st.button("Launch KPI", use_container_width=True): st.session_state['current_app']='KPI'; st.rerun()
    with c2:
        with st.container(border=True):
            st.markdown("### 🎓 **Training**"); st.caption("Track Progress")
            if st.button("Launch Training", use_container_width=True): st.session_state['current_app']='TRAINING'; st.rerun()
    with c3:
        with st.container(border=True):
            st.markdown("### 🚀 **Tracker**"); st.caption("HR & Finance")
            if st.session_state['role'] in ['Team Leader', 'Super Admin']:
                if st.button("Launch Tracker", use_container_width=True): st.session_state['current_app']='RESOURCE'; st.rerun()
            else:
                st.button("Restricted", disabled=True, use_container_width=True)

def parse_date(d):
    if not d or d == 'None' or d == '': return None
    try: return pd.to_datetime(d).date()
    except: return None

# --- NEW APP SECTION: MY PROFILE ---
def app_my_profile():
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("⬅ Home", use_container_width=True): st.session_state['current_app']='HOME'; st.rerun()
    with c2: st.markdown("### 👤 My Profile & Settings")
    st.markdown("---")

    tab_det, tab_set = st.tabs(["📄 My Details", "⚙️ Account Settings"])

    # 1. MY DETAILS TAB (Resource Info without costs)
    with tab_det:
        my_id = st.session_state.get('emp_id')
        if not my_id:
            st.warning("No Employee ID linked to your account. Contact Admin.")
        else:
            df = get_user_resource_details(my_id)
            if not df.empty:
                data = df.iloc[0]
                
                # Layout
                c_prof, c_info = st.columns([1, 3])
                with c_prof:
                    st.image(st.session_state.get('img'), width=150)
                    st.markdown(f"**{data['employee_name']}**")
                    st.caption(f"{data['department']} | {data['location']}")
                    if data['status'] == 'Active':
                        st.success("Status: Active")
                    else:
                        st.error(f"Status: {data['status']}")

                with c_info:
                    with st.container(border=True):
                        st.subheader("Official Details")
                        # Exclude sensitive fields: hourly_rate, hardware_daily_cost, po_details
                        ic1, ic2 = st.columns(2)
                        with ic1:
                            st.text_input("Employee ID", value=data['employee_id'], disabled=True)
                            st.text_input("Reporting Manager", value=data['reporting_manager'], disabled=True)
                            st.text_input("Experience Level", value=data['experience_level'], disabled=True)
                        with ic2:
                            st.text_input("DEV Code", value=data['dev_code'], disabled=True)
                            st.text_input("Onboarding Date", value=data['onboarding_date'], disabled=True)
                            st.text_input("Department", value=data['department'], disabled=True)
                        
                        st.markdown("**Remarks:**")
                        st.info(data['remarks'] if data['remarks'] else "No remarks.")
            else:
                st.info(f"No resource record found for Employee ID: {my_id}. Please ask your lead to update the Resource Tracker.")

    # 2. ACCOUNT SETTINGS TAB
    with tab_set:
        c_pass, c_photo = st.columns(2)
        
        with c_pass:
            with st.container(border=True):
                st.subheader("🔐 Change Password")
                curr_pass = st.text_input("Current Password", type="password")
                new_pass = st.text_input("New Password", type="password")
                conf_pass = st.text_input("Confirm New Password", type="password")
                
                if st.button("Update Password", type="primary", use_container_width=True):
                    # Verify current password
                    conn = sqlite3.connect(DB_FILE)
                    cur = conn.cursor()
                    cur.execute("SELECT password FROM users WHERE username=?", (st.session_state['user'],))
                    db_pass = cur.fetchone()[0]
                    conn.close()
                    
                    if curr_pass != db_pass:
                        st.error("Current password incorrect.")
                    elif new_pass != conf_pass:
                        st.error("New passwords do not match.")
                    elif not new_pass:
                        st.error("Password cannot be empty.")
                    else:
                        update_user_credentials(st.session_state['user'], new_password=new_pass)
                        st.success("Password updated successfully! Please login again.")
                        st.session_state['logged_in'] = False # Force re-login
                        st.rerun()

        with c_photo:
            with st.container(border=True):
                st.subheader("🖼️ Change Profile Photo")
                st.write("Current URL:")
                st.caption(st.session_state.get('img'))
                new_img = st.text_input("New Image URL", placeholder="https://example.com/my-photo.png")
                
                if st.button("Update Photo", use_container_width=True):
                    if new_img:
                        update_user_credentials(st.session_state['user'], new_img=new_img)
                        st.session_state['img'] = new_img # Update session immediately
                        st.success("Profile photo updated!")
                        st.rerun()
                    else:
                        st.error("Please enter a valid URL.")


# --- ADMIN APP ---
def app_admin():
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("⬅ Home", use_container_width=True): st.session_state['current_app']='HOME'; st.rerun()
    with c2: st.markdown("### 🛡️ Super Admin Control Panel")
    st.markdown("---")

    if 'admin_mode' not in st.session_state: st.session_state['admin_mode'] = 'TABLE'
    if 'admin_edit_user' not in st.session_state: st.session_state['admin_edit_user'] = None

    t1, t2 = st.tabs(["👥 User Management", "📥 Import/Export"])

    with t1:
        if st.session_state['admin_mode'] == 'TABLE':
            col_act, col_add = st.columns([5, 1])
            with col_act: st.info("Manage portal access, reset passwords, and assign roles.")
            with col_add: 
                if st.button("➕ New User", type="primary", use_container_width=True):
                    st.session_state['admin_mode'] = 'FORM'
                    st.session_state['admin_edit_user'] = None
                    st.rerun()
            
            df = get_all_users()
            display_df = df.drop(columns=['password'])
            edited_df = st.data_editor(display_df, use_container_width=True, num_rows="dynamic", key="user_editor")
            
            st.caption("Select a user from the dropdown below to Edit fully or Reset Password.")
            ac1, ac2, ac3 = st.columns([2, 1, 1])
            with ac1:
                sel_user = st.selectbox("Select User to Modify", df['username'], label_visibility="collapsed")
            with ac2:
                if st.button("✏️ Edit / Reset Pass", use_container_width=True):
                    st.session_state['admin_edit_user'] = sel_user
                    st.session_state['admin_mode'] = 'FORM'
                    st.rerun()
            with ac3:
                if st.button("🗑️ Delete", type="primary", use_container_width=True):
                    if sel_user == 'admin': st.error("Cannot delete Super Admin.")
                    else:
                        delete_user(sel_user)
                        st.success(f"User {sel_user} deleted."); st.rerun()

        elif st.session_state['admin_mode'] == 'FORM':
            st.subheader("User Account Details")
            is_edit = st.session_state['admin_edit_user'] is not None
            u_data = {}
            if is_edit:
                df = get_all_users()
                u_data = df[df['username'] == st.session_state['admin_edit_user']].iloc[0].to_dict()
            
            with st.container(border=True):
                f1, f2 = st.columns(2)
                with f1:
                    username = st.text_input("Username (Login ID)", value=u_data.get('username',''), disabled=is_edit)
                    name = st.text_input("Full Name", value=u_data.get('name',''))
                    role = st.selectbox("Role", ["Team Member", "Team Leader", "Super Admin"], 
                                        index=["Team Member", "Team Leader", "Super Admin"].index(u_data.get('role','Team Member')))
                with f2:
                    emp_id = st.text_input("Employee ID Link", value=u_data.get('emp_id',''))
                    if not is_edit:
                        temp_pass = generate_temp_password()
                        password = st.text_input("Password (Auto-Generated Temp)", value=temp_pass)
                        st.info(f"📝 Note this temporary password: **{password}**")
                    else:
                        password = st.text_input("Reset Password (Leave as is to keep current)", value=u_data.get('password',''))
                    img = st.text_input("Profile Image URL", value=u_data.get('img','https://cdn-icons-png.flaticon.com/512/3135/3135715.png'))

                st.markdown("<br>", unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("Cancel", use_container_width=True):
                        st.session_state['admin_mode'] = 'TABLE'; st.rerun()
                with b2:
                    if st.button("💾 Save User", type="primary", use_container_width=True):
                        if not username or not password:
                            st.error("Username and Password are required.")
                        else:
                            payload = {'username': username, 'password': password, 'role': role, 'name': name, 'emp_id': emp_id, 'img': img}
                            save_user_entry(payload, is_update=is_edit)
                            st.success("User saved successfully!")
                            st.session_state['admin_mode'] = 'TABLE'
                            st.rerun()

    with t2:
        st.subheader("Bulk Operations")
        c_imp, c_exp = st.columns(2)
        with c_imp:
            up_users = st.file_uploader("Import Users (CSV)", type=['csv'])
            if up_users:
                if import_users_csv(up_users): st.success("Users Imported!"); st.rerun()
        with c_exp:
            df_exp = get_all_users()
            st.download_button("Download User Database (CSV)", data=df_exp.to_csv(index=False).encode('utf-8'), file_name="portal_users.csv", mime="text/csv", use_container_width=True)

# --- FULL KPI APP ---
def app_kpi():
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("⬅ Home", use_container_width=True): st.session_state['current_app']='HOME'; st.rerun()
    with c2: st.markdown("### 📊 KPI Management System")
    st.markdown("---")
    
    is_lead = st.session_state['role'] in ["Team Leader", "Super Admin"]
    
    if is_lead:
        df = get_kpi_data()
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
                    u_df = get_all_users()
                    pilots = u_df[u_df['role'] == "Team Member"]['name'].tolist()
                    if not pilots: pilots = ["Generic Pilot"]
                    
                    with c1:
                        tname = st.text_input("Task Name", value=default_data.get("task_name", ""))
                        pilot_val = default_data.get("name_activity_pilot")
                        p_idx = pilots.index(pilot_val) if pilot_val in pilots else 0
                        pilot = st.selectbox("Assign To", pilots, index=p_idx)
                        otd_curr = default_data.get("otd_customer", "N/A")
                        st.text_input("OTD Status (Computed)", value=otd_curr, disabled=True)
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
                        ftr_opts = ["Yes", "No", "Awaited"]
                        ftr_val = default_data.get("ftr_internal", "Yes")
                        ftr = st.selectbox("FTR Internal", ftr_opts, index=ftr_opts.index(ftr_val) if ftr_val in ftr_opts else 0)
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
                            st_color = "black"
                            if row['status'] == "Completed": st_color = "#10b981"
                            elif row['status'] == "Cancelled": st_color = "#ef4444"
                            elif row['status'] == "Hold": st_color = "#f59e0b"
                            else: st_color = "#3b82f6"
                            st.markdown(f"<span style='color:{st_color}; font-weight:bold;'>{row['status']}</span> | OTD: {row.get('otd_customer','-')}", unsafe_allow_html=True)
                        with c_btn:
                            if st.button("Edit", key=f"kpi_edit_{row['id']}", use_container_width=True):
                                st.session_state['edit_kpi_id'] = row['id']; st.rerun()
            else: st.info("No tasks found.")

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
    
    if st.session_state['role'] in ["Team Leader", "Super Admin"]:
        t1, t2 = st.tabs(["Repository", "Add New"])
        with t1:
            df = get_trainings()
            
            with st.expander("📂 Import / Export", expanded=True):
                col_imp, col_exp = st.columns(2)
                with col_imp:
                    up_train = st.file_uploader("Upload CSV", type=['csv'], key="train_csv_up")
                    if up_train:
                        if import_training_csv(up_train): st.success("Trainings Imported!"); st.rerun()
                with col_exp:
                    if not df.empty:
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download CSV", data=csv, file_name="training_repo.csv", mime="text/csv", use_container_width=True)
                    else:
                        tpl = pd.DataFrame(columns=['title', 'description', 'link', 'role_target', 'mandatory'])
                        csv = tpl.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Template CSV", data=csv, file_name="training_template.csv", mime="text/csv", use_container_width=True)

            if not df.empty:
                st.markdown("#### Manage Modules")
                df_editor = df.copy()
                df_editor.insert(0, "Select", False)
                
                edited_df = st.data_editor(
                    df_editor,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Select": st.column_config.CheckboxColumn("Select", help="Select to delete"),
                        "id": st.column_config.TextColumn("ID", disabled=True),
                        "link": st.column_config.LinkColumn("Link"),
                        "created_by": st.column_config.TextColumn("Creator", disabled=True)
                    }
                )
                
                col_del_sel, col_del_all = st.columns([1, 1])
                with col_del_sel:
                    if st.button("🗑️ Delete Selected", type="primary"):
                        to_delete = edited_df[edited_df['Select'] == True]
                        if not to_delete.empty:
                            for _, row in to_delete.iterrows():
                                delete_training(row['id'])
                            st.success(f"Deleted {len(to_delete)} modules.")
                            st.rerun()
                        else:
                            st.warning("Select items to delete first.")
                with col_del_all:
                    if st.button("⚠️ DELETE ALL", type="primary"):
                        delete_all_trainings()
                        st.rerun()
            else: 
                st.info("Repository empty.")

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

# --- RESOURCE TRACKER APP ---
def app_resource():
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("⬅ Home", use_container_width=True): st.session_state['current_app']='HOME'; st.rerun()
    with c2: st.markdown("### 🚀 Resource Tracker")
    st.markdown("---")

    if st.session_state['role'] not in ["Team Leader", "Super Admin"]:
        st.error("🚫 ACCESS RESTRICTED")
        return

    if 'res_edit_id' not in st.session_state: st.session_state['res_edit_id'] = None
    if 'res_view_mode' not in st.session_state: st.session_state['res_view_mode'] = 'LIST' 

    if st.session_state['res_view_mode'] == 'LIST':
        with st.expander("📂 Import / Export", expanded=False):
            rc1, rc2 = st.columns(2)
            with rc1:
                up_res = st.file_uploader("Import Resource CSV", type=['csv'], key="res_csv_up")
                if up_res:
                    if import_resource_csv(up_res): st.success("Resources Imported!"); st.rerun()
            with rc2:
                df_all = get_resource_list()
                if not df_all.empty:
                    csv_res = df_all.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Data CSV", data=csv_res, file_name="resources.csv", mime="text/csv", use_container_width=True)
                else:
                    st.info("No data to export.")

        with st.expander("🔎 Search & Filters", expanded=False):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                search_query = st.text_input("Search Name/ID/Dev", placeholder="Type to search...")
            with fc2:
                dept_filter = st.multiselect("Filter Department", ["Engineering", "Quality", "Manufacturing"])
            with fc3:
                stat_filter = st.multiselect("Filter Status", ["Active", "Inactive", "Yet to start"])

        col_act, col_add = st.columns([5, 1])
        with col_act:
            st.markdown("#### Resource List")
        with col_add:
            if st.button("➕ Add New", type="primary", use_container_width=True):
                st.session_state['res_edit_id'] = None
                st.session_state['res_view_mode'] = 'FORM'
                st.rerun()
        
        df = get_resource_list()
        
        if not df.empty:
            if search_query:
                df = df[df.apply(lambda row: search_query.lower() in str(row.values).lower(), axis=1)]
            if dept_filter:
                df = df[df['department'].isin(dept_filter)]
            if stat_filter:
                df = df[df['status'].isin(stat_filter)]
            
            df['hourly_rate'] = pd.to_numeric(df['hourly_rate'], errors='coerce').fillna(0)
            df['hardware_daily_cost'] = pd.to_numeric(df['hardware_daily_cost'], errors='coerce').fillna(0)
            df['Daily_Labor_Cost_$'] = df['hourly_rate'] * 8
            df['Total_Daily_Bill_$'] = df['Daily_Labor_Cost_$'] + df['hardware_daily_cost']
            
            display_cols = ['employee_name', 'employee_id', 'department', 'status', 'location', 
                            'hourly_rate', 'Daily_Labor_Cost_$', 'Total_Daily_Bill_$']
            
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
            
            st.markdown("##### Manage Entry")
            if len(df) > 0:
                sel_res = st.selectbox("Select Resource to Edit/View", df['employee_name'] + " (" + df['employee_id'] + ")")
                if st.button("Edit Selected", use_container_width=True):
                    sel_id = df[df['employee_name'] + " (" + df['employee_id'] + ")" == sel_res].iloc[0]['id']
                    st.session_state['res_edit_id'] = sel_id
                    st.session_state['res_view_mode'] = 'FORM'
                    st.rerun()
            else:
                st.info("No records match your filters.")
        else:
            st.info("No resources found in database.")

    elif st.session_state['res_view_mode'] == 'FORM':
        res_id = st.session_state['res_edit_id']
        is_edit = res_id is not None
        
        st.subheader("Edit Resource" if is_edit else "New Resource Onboarding")
        d = {}
        if is_edit:
            df = get_resource_list()
            row = df[df['id'] == res_id]
            if not row.empty: d = row.iloc[0].to_dict()

        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                emp_name = st.text_input("Employee Name", value=d.get('employee_name', ''))
                dev_opts = ["001", "002", "003", "005", "012", "016", "089"]
                dev_val = d.get('dev_code', '001')
                dev_code = st.selectbox("DEV", dev_opts, index=dev_opts.index(dev_val) if dev_val in dev_opts else 0)
                loc_opts = ["Chennai", "Bangalore", "Pune", "Remote"]
                loc_val = d.get('location', 'Chennai')
                location = st.selectbox("Location", loc_opts, index=loc_opts.index(loc_val) if loc_val in loc_opts else 0)
                o_date = st.date_input("Onboarding Start Date", value=parse_date(d.get('onboarding_date')) or date.today())
                stat_opts = ["Yet to start", "Active", "Inactive"]
                stat_val = d.get('status', 'Yet to start')
                status = st.selectbox("Status", stat_opts, index=stat_opts.index(stat_val) if stat_val in stat_opts else 0)

            with col2:
                emp_id_val = st.text_input("Employee ID", value=d.get('employee_id', ''))
                dept_opts = ["Engineering", "Quality", "Manufacturing"]
                dept_val = d.get('department', 'Engineering')
                department = st.selectbox("Department", dept_opts, index=dept_opts.index(dept_val) if dept_val in dept_opts else 0)
                rep_man = st.selectbox("Reporting Manager", ["Sarah Jenkins", "Mike Ross", "Harvey Specter"], index=0)
                exp_opts = ["JUNIOR", "MID", "ADVANCED", "SENIOR", "EXPERT"]
                exp_val = d.get('experience_level', 'JUNIOR')
                exp_lvl = st.selectbox("Experience Level", exp_opts, index=exp_opts.index(exp_val) if exp_val in exp_opts else 0)
                po_det = st.text_input("PO Details", value=d.get('po_details', ''), placeholder="PO number")

            st.markdown("##### 💲 Financials (USD)")
            fin1, fin2, fin3, fin4 = st.columns(4)
            with fin1:
                hr_rate = st.number_input("Hourly Rate ($)", min_value=0.0, value=float(d.get('hourly_rate', 0.0)))
            with fin2:
                hw_cost = st.number_input("Hardware Cost (Daily $)", min_value=0.0, value=float(d.get('hardware_daily_cost', 0.0)))
            with fin3:
                lab_daily = hr_rate * 8
                st.metric("Labor Daily (8h)", f"${lab_daily:,.2f}")
            with fin4:
                tot_daily = lab_daily + hw_cost
                st.metric("Total Daily Bill", f"${tot_daily:,.2f}")

            remarks = st.text_area("Remarks if any", value=d.get('remarks', ''))
            exit_date, backfill, reason = None, "No", ""
            
            if status == "Inactive":
                st.markdown("---")
                st.warning("⚠️ Resource Inactive - Exit Details Required")
                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    exit_date = st.date_input("Effective Exit Date", value=parse_date(d.get('effective_exit_date')) or date.today())
                with ec2:
                    bf_val = d.get('backfill_status', 'No')
                    backfill = st.selectbox("Backfill Status", ["Yes", "No"], index=["Yes", "No"].index(bf_val) if bf_val in ["Yes", "No"] else 1)
                with ec3:
                    reason = st.text_input("Reason for Leaving", value=d.get('reason_for_leaving', ''))

            st.markdown("<br>", unsafe_allow_html=True)
            b1, b2 = st.columns([1, 1])
            with b1:
                if st.button("Cancel", use_container_width=True):
                    st.session_state['res_view_mode'] = 'LIST'
                    st.session_state['res_edit_id'] = None
                    st.rerun()
            with b2:
                if st.button("💾 Save Record", type="primary", use_container_width=True):
                    if not emp_name or not emp_id_val:
                        st.error("Name and Employee ID are required.")
                    elif status == "Inactive" and not reason:
                        st.error("Reason for Leaving is mandatory for Inactive status.")
                    else:
                        payload = {
                            "employee_name": emp_name, "employee_id": emp_id_val, "dev_code": dev_code,
                            "department": department, "location": location, "reporting_manager": rep_man,
                            "onboarding_date": str(o_date), "experience_level": exp_lvl, "status": status,
                            "po_details": po_det, "remarks": remarks,
                            "effective_exit_date": str(exit_date) if exit_date else "",
                            "backfill_status": backfill if status == "Inactive" else "",
                            "reason_for_leaving": reason if status == "Inactive" else "",
                            "hourly_rate": str(hr_rate), "hardware_daily_cost": str(hw_cost)
                        }
                        temp_pass = save_resource_entry(payload, res_id)
                        if temp_pass:
                            st.success(f"✅ Resource Added! Login created: {emp_id_val.lower().replace(' ','')} | Pass: {temp_pass}")
                        else:
                            st.success("Resource Saved Successfully!")
                        st.session_state['res_view_mode'] = 'LIST'
                        st.session_state['res_edit_id'] = None
                        # No st.rerun() here so user can see the temp password toast

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
            
            # --- NEW SIDEBAR LINK ---
            if st.button("👤 My Profile", use_container_width=True): 
                st.session_state['current_app'] = 'MY_PROFILE'
                st.rerun()
            
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
        elif app == 'ADMIN': app_admin()
        elif app == 'MY_PROFILE': app_my_profile() # --- NEW ROUTE ---

if __name__ == "__main__":
    main()
