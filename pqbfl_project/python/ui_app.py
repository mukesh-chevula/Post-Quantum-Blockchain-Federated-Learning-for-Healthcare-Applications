from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import altair as alt
import pandas as pd
import streamlit as st
from web3 import Web3

from pqbfl.scripts.demo_end_to_end import DemoConfig, run_demo


@dataclass
class NodeState:
    proc: subprocess.Popen
    log_path: Path


def _project_root() -> Path:
    # .../pqbfl_project/python/ui_app.py -> .../pqbfl_project
    return Path(__file__).resolve().parents[1]


def _chain_dir() -> Path:
    return _project_root() / "chain"


def _python_dir() -> Path:
    return _project_root() / "python"


def _default_chain_url() -> str:
    return os.getenv("PQBFL_CHAIN_URL", "http://127.0.0.1:8545")


def _parse_chain_url(url: str) -> tuple[str, int]:
    # Accept both full URLs (http://127.0.0.1:8545) and host:port.
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8545
    return host, port


def _read_tail(path: Path, n: int = 120) -> str:
    if not path.exists():
        return "(no log yet)"
    try:
        lines = path.read_text(errors="ignore").splitlines()
        return "\n".join(lines[-n:])
    except Exception as e:
        return f"(failed to read log: {e})"


def _node_running(url: str) -> bool:
    try:
        w3 = Web3(Web3.HTTPProvider(url))
        return bool(w3.is_connected())
    except Exception:
        return False


def _ensure_session_state():
    if "node" not in st.session_state:
        st.session_state.node = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "chain_url" not in st.session_state:
        st.session_state.chain_url = _default_chain_url()


def _port_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def _find_free_port(host: str, preferred_port: int, tries: int = 20) -> int | None:
    for p in range(preferred_port, preferred_port + tries):
        if _port_available(host, p):
            return p
    return None


def start_hardhat_node(url: str) -> None:
    if _node_running(url):
        st.info("Hardhat node already reachable; not starting another one.")
        return

    chain_dir = _chain_dir()
    log_path = Path("/tmp/pqbfl_hardhat_ui.log")

    host, port = _parse_chain_url(url)
    if not _port_available(host, port):
        alt_port = _find_free_port(host, port + 1)
        if alt_port is None:
            st.error(
                f"Port {port} is not available and no free port was found nearby. "
                "Stop the existing process using the port, or choose another Chain URL port."
            )
            return
        st.warning(f"Port {port} is busy. Starting Hardhat on port {alt_port} instead.")
        port = alt_port
        url = f"http://{host}:{port}"
        st.session_state.chain_url = url

    env = os.environ.copy()
    env.setdefault("HARDHAT_DISABLE_TELEMETRY", "1")
    env.setdefault("CI", "1")

    # Hardhat is typically installed in the chain folder via npm install.
    cmd = ["npx", "hardhat", "node", "--hostname", host, "--port", str(port)]

    with log_path.open("w") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(chain_dir),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    st.session_state.node = NodeState(proc=proc, log_path=log_path)

    # Give it a moment to bind.
    for _ in range(20):
        if proc.poll() is not None:
            break
        if _node_running(url):
            return
        time.sleep(0.25)

    if not _node_running(url):
        st.session_state.node = None
        st.error(
            "Failed to start Hardhat node. The port may already be in use. "
            "Try stopping the other node, or change the Chain URL port (e.g., 8546)."
        )


def stop_hardhat_node() -> None:
    node: NodeState | None = st.session_state.node
    if node is None:
        return

    if node.proc.poll() is not None:
        st.session_state.node = None
        return

    try:
        # We started with start_new_session=True, so PID is a process group leader.
        os.killpg(node.proc.pid, signal.SIGTERM)
        node.proc.wait(timeout=4)
    except Exception:
        try:
            os.killpg(node.proc.pid, signal.SIGKILL)
        except Exception:
            try:
                node.proc.kill()
            except Exception:
                pass

    st.session_state.node = None


def main():
    _ensure_session_state()

    st.set_page_config(page_title="PQBFL UI", layout="wide")

    st.title("PQBFL: Post-Quantum Blockchain FL")
    st.caption("Local Hardhat chain + on/off-chain PQBFL flow + encrypted FL rounds")

    with st.sidebar:
        st.header("Run settings")
        chain_url = st.text_input(
            "Chain URL",
            key="chain_url",
            help="Hardhat JSON-RPC endpoint",
        )
        rounds = st.number_input("Rounds", min_value=1, max_value=50, value=int(os.getenv("PQBFL_ROUNDS", "6")))
        clients = st.number_input("Clients", min_value=1, max_value=10, value=int(os.getenv("PQBFL_CLIENTS", "2")))
        L_j = st.number_input("L_j (symmetric ratchet window)", min_value=1, max_value=50, value=int(os.getenv("PQBFL_LJ", "3")))
        project_id = st.number_input("Project ID", min_value=1, max_value=1_000_000, value=int(os.getenv("PQBFL_PROJECT_ID", "1")))
        st.divider()

        st.subheader("Chain")
        st.code(chain_url)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Start node", width="stretch"):
                start_hardhat_node(chain_url)
        with col_b:
            if st.button("Stop node", width="stretch"):
                stop_hardhat_node()

        st.write("Node reachable:", "✅" if _node_running(chain_url) else "❌")

    

    st.subheader("Demo execution")

    if not _node_running(chain_url):
        st.warning("Hardhat node is not reachable. Start it first (or run `npm run node` in pqbfl_project/chain`).")

    if st.button("Run demo", type="primary", width="stretch", disabled=not _node_running(chain_url)):
        cfg = DemoConfig(
            chain_url=chain_url,
            rounds=int(rounds),
            n_clients=int(clients),
            L_j=int(L_j),
            project_id=int(project_id),
        )
        with st.spinner("Running PQBFL demo..."):
            st.session_state.last_result = run_demo(cfg)

    result = st.session_state.last_result
    if result is not None:
        st.success("✅ Run completed successfully!")
        
        # Main metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Contract Address", result.contract_address[:10] + "...")
        with col2:
            st.metric("Initial Accuracy", f"{result.initial_accuracy:.2%}")
        with col3:
            st.metric("Final Accuracy", f"{result.final_accuracy:.2%}")

    if st.session_state.last_result is not None:
        st.subheader("📊 Accuracy over rounds")
        r = st.session_state.last_result
        df = pd.DataFrame({"round": list(range(0, len(r.round_accuracies))), "accuracy": r.round_accuracies})
        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(x=alt.X("round:Q", title="Round"), y=alt.Y("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])))
            .properties(height=320, title="Model Accuracy Over FL Rounds")
        )
        st.altair_chart(chart, use_container_width=True)
        
        # Transaction timing section
        st.markdown("---")
        st.subheader("⏱️ Real-Time Blockchain Transaction Timings")
        
        # Transaction statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", r.total_transactions)
        with col2:
            st.metric("Avg Time (ms)", f"{r.avg_transaction_time_ms:.2f}")
        with col3:
            st.metric("Min Time (ms)", f"{r.min_transaction_time_ms:.2f}")
        with col4:
            st.metric("Max Time (ms)", f"{r.max_transaction_time_ms:.2f}")
        
        # Display all transactions
        if r.transaction_timings:
            tx_df = pd.DataFrame(r.transaction_timings)
            
            # Format for display
            display_df = tx_df.copy()
            display_df["tx_hash"] = display_df["tx_hash"].str[:10] + "..."
            display_df["round"] = display_df["round"].astype(str)
            display_df["client"] = display_df["client"].apply(lambda x: "Server" if x == -1 else f"Client {x}")
            display_df = display_df[["tx_type", "round", "client", "duration_ms", "gas_used"]]
            display_df.columns = ["Transaction Type", "Round", "Participant", "Duration (ms)", "Gas Used"]
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Transaction type breakdown
            st.markdown("### 📈 Transaction Type Analysis")
            type_stats = tx_df.groupby("tx_type").agg({
                "duration_ms": ["count", "mean", "min", "max", "std"],
                "gas_used": ["mean", "max"]
            }).round(2)
            type_stats.columns = ["Count", "Avg Time (ms)", "Min Time (ms)", "Max Time (ms)", "Std Dev (ms)", "Avg Gas", "Max Gas"]
            st.dataframe(type_stats, use_container_width=True)
            
            # Timing visualization by transaction type
            fig_data = tx_df[["tx_type", "duration_ms"]].copy()
            fig_data["transaction_number"] = range(1, len(fig_data) + 1)
            
            timing_chart = (
                alt.Chart(fig_data)
                .mark_circle(size=60)
                .encode(
                    x=alt.X("transaction_number:Q", title="Transaction Number"),
                    y=alt.Y("duration_ms:Q", title="Duration (ms)"),
                    color=alt.Color("tx_type:N", title="Transaction Type"),
                    tooltip=["transaction_number", "tx_type", "duration_ms"]
                )
                .properties(height=300, title="Transaction Timing by Type")
            )
            st.altair_chart(timing_chart, use_container_width=True)
        else:
            st.info("No transaction timing data available")

        # Off-chain operation timing section
        st.markdown("---")
        st.subheader("⚙️ Off-Chain Operation Timings")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Operations", r.total_operations)
        with col2:
            st.metric("Avg Time (ms)", f"{r.avg_operation_time_ms:.2f}")
        with col3:
            st.metric("Min Time (ms)", f"{r.min_operation_time_ms:.2f}")
        with col4:
            st.metric("Max Time (ms)", f"{r.max_operation_time_ms:.2f}")

        if getattr(r, "operation_timings", None):
            op_df = pd.DataFrame(r.operation_timings)

            display_ops = op_df.copy()
            display_ops["round"] = display_ops["round"].astype(str)
            display_ops["client"] = display_ops["client"].apply(lambda x: "Server" if x == -1 else f"Client {x}")
            display_ops = display_ops[["op_type", "round", "client", "duration_ms"]]
            display_ops.columns = ["Operation", "Round", "Participant", "Duration (ms)"]

            st.dataframe(display_ops, use_container_width=True, height=400)

            st.markdown("### 🔍 Operation Type Analysis")
            op_stats = op_df.groupby("op_type").agg({
                "duration_ms": ["count", "mean", "min", "max", "std"]
            }).round(2)
            op_stats.columns = ["Count", "Avg Time (ms)", "Min Time (ms)", "Max Time (ms)", "Std Dev (ms)"]
            st.dataframe(op_stats, use_container_width=True)

            op_fig = op_df[["op_type", "duration_ms"]].copy()
            op_fig["operation_number"] = range(1, len(op_fig) + 1)
            op_chart = (
                alt.Chart(op_fig)
                .mark_circle(size=60)
                .encode(
                    x=alt.X("operation_number:Q", title="Operation Number"),
                    y=alt.Y("duration_ms:Q", title="Duration (ms)"),
                    color=alt.Color("op_type:N", title="Operation Type"),
                    tooltip=["operation_number", "op_type", "duration_ms"]
                )
                .properties(height=300, title="Operation Timing by Type")
            )
            st.altair_chart(op_chart, use_container_width=True)
        else:
            st.info("No off-chain operation timing data available")

        with st.expander("Raw result"):
            st.json(r.as_dict())


if __name__ == "__main__":
    main()
