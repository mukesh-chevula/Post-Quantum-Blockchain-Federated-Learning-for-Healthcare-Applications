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

    st.title("PQBFL: Post-Quantum Blockchain FL — Demo UI")
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
        st.subheader("FL / data")
        non_iid = st.toggle("Non-IID clients", value=os.getenv("PQBFL_NON_IID", "1") not in ("0", "false", "False"))
        data_seed = st.number_input("Dataset seed", min_value=0, max_value=1_000_000, value=int(os.getenv("PQBFL_DATA_SEED", "42")))
        model_seed = st.number_input("Model init seed", min_value=0, max_value=1_000_000, value=int(os.getenv("PQBFL_MODEL_SEED", "0")))

        lr = st.number_input("Learning rate", min_value=0.001, max_value=5.0, value=float(os.getenv("PQBFL_LR", "0.2")))
        epochs = st.number_input("Local epochs", min_value=1, max_value=50, value=int(os.getenv("PQBFL_EPOCHS", "2")))
        batch_size = st.number_input("Batch size", min_value=8, max_value=4096, value=int(os.getenv("PQBFL_BATCH_SIZE", "64")))
        l2 = st.number_input("L2 regularization", min_value=0.0, max_value=10.0, value=float(os.getenv("PQBFL_L2", "0.0")))

        st.divider()
        st.subheader("Simulation")
        sim_seed = st.number_input("Simulation seed", min_value=0, max_value=1_000_000, value=int(os.getenv("PQBFL_SIM_SEED", "123")))
        participation_rate = st.slider("Client participation rate", min_value=0.1, max_value=1.0, value=float(os.getenv("PQBFL_PARTICIPATION", "1.0")))
        label_flip_prob = st.slider("Label flip prob (poisoning)", min_value=0.0, max_value=0.5, value=float(os.getenv("PQBFL_LABEL_FLIP_PROB", "0.0")))

        st.divider()
        st.subheader("Aggregation")
        aggregator = st.selectbox("Aggregator", options=["fedavg", "median", "trimmed_mean"], index=0)
        trim_ratio = 0.1
        if aggregator == "trimmed_mean":
            trim_ratio = st.slider("Trim ratio", min_value=0.0, max_value=0.49, value=float(os.getenv("PQBFL_TRIM_RATIO", "0.1")))
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

            non_iid=bool(non_iid),
            data_seed=int(data_seed),
            model_seed=int(model_seed),
            lr=float(lr),
            epochs=int(epochs),
            batch_size=int(batch_size),
            l2=float(l2),

            sim_seed=int(sim_seed),
            participation_rate=float(participation_rate),
            label_flip_prob=float(label_flip_prob),

            aggregator=str(aggregator),
            trim_ratio=float(trim_ratio),
        )
        with st.spinner("Running PQBFL demo..."):
            st.session_state.last_result = run_demo(cfg)

    result = st.session_state.last_result
    if result is not None:
        st.success("Run completed")
        st.write("Contract:", result.contract_address)
        st.write("Initial accuracy:", f"{result.initial_accuracy:.4f}")
        st.write("Final accuracy:", f"{result.final_accuracy:.4f}")

    if st.session_state.last_result is not None:
        st.subheader("Accuracy over rounds")
        r = st.session_state.last_result
        df = pd.DataFrame({"round": list(range(0, len(r.round_accuracies))), "accuracy": r.round_accuracies})
        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(x=alt.X("round:Q", title="Round"), y=alt.Y("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])))
            .properties(height=320)
        )
        st.altair_chart(chart, width="stretch")

        with st.expander("Raw result"):
            st.json(r.as_dict())


if __name__ == "__main__":
    main()
