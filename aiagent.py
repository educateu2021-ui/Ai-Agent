import streamlit as st
import json
import requests
from typing import Dict, Any

st.set_page_config(layout="wide")

# -------------------- NODE REGISTRY --------------------
NODE_TYPES = {
    "Webhook Trigger": {"type": "trigger"},
    "HTTP Request": {"type": "action"},
    "Python Function": {"type": "action"},
}

WORKFLOW_STORE = "workflows.json"


# -------------------- Load/Save Workflows --------------------
def load_workflows():
    try:
        with open(WORKFLOW_STORE, "r") as f:
            return json.load(f)
    except:
        return {}


def save_workflows(data):
    with open(WORKFLOW_STORE, "w") as f:
        json.dump(data, f, indent=4)


# -------------------- Execute Workflow --------------------
def execute_node(node: Dict[str, Any], payload=None):
    if node["type"] == "Webhook Trigger":
        return payload

    if node["type"] == "HTTP Request":
        config = node["config"]
        response = requests.request(
            method=config["method"],
            url=config["url"],
            json=config.get("body", {})
        )
        return response.json()

    if node["type"] == "Python Function":
        code = node["config"]["code"]
        exec_globals = {}
        exec(code, exec_globals)
        return exec_globals["handler"](payload)

    return None


def run_workflow(workflow_json, incoming_payload=None):
    output = incoming_payload
    for node in workflow_json["nodes"]:
        output = execute_node(node, payload=output)
    return output


# -------------------- UI --------------------
st.title("🔗 Workflow Builder (n8n style)")

tab1, tab2, tab3 = st.tabs(["🧱 Build Workflow", "⚡ Execute Workflow", "📂 Saved Workflows"])

# -------------------- TAB 1 - Builder --------------------
with tab1:
    st.subheader("Create Workflow")

    workflow_name = st.text_input("Workflow Name")
    node_list = []

    st.markdown("### Add Nodes")
    for i in range(1, 6):
        st.markdown(f"#### Node {i}")
        node_type = st.selectbox(f"Select Type for Node {i}", ["None"] + list(NODE_TYPES.keys()), key=f"node_{i}")

        if node_type != "None":
            node = {"id": f"node{i}", "type": node_type, "config": {}}

            # Capture config for nodes
            if node_type == "HTTP Request":
                node["config"]["url"] = st.text_input(f"URL for Node {i}", "")
                node["config"]["method"] = st.selectbox(f"Method for Node {i}", ["GET", "POST"])
                node["config"]["body"] = st.text_area(f"Body JSON (Node {i})", "{}")

            if node_type == "Python Function":
                node["config"]["code"] = st.text_area(
                    f"Handler Function (Node {i})", 
                    "def handler(payload):\n    return {'message': 'Hello from Python', 'input': payload}"
                )

            node_list.append(node)

    if st.button("Save Workflow"):
        workflows = load_workflows()
        workflows[workflow_name] = {"name": workflow_name, "nodes": node_list}
        save_workflows(workflows)
        st.success("Workflow Saved!")


# -------------------- TAB 2 - Execute --------------------
with tab2:
    workflows = load_workflows()
    selected = st.selectbox("Select Workflow", list(workflows.keys()))

    sample_payload = st.text_area("Incoming Payload (JSON)", "{}")

    if st.button("Run Workflow"):
        payload = json.loads(sample_payload)
        result = run_workflow(workflows[selected], payload)
        st.json(result)


# -------------------- TAB 3 - Saved Workflows --------------------
with tab3:
    st.subheader("Saved Workflows")
    workflows = load_workflows()
    st.json(workflows)
