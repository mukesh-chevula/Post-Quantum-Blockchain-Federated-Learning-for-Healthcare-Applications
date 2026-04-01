import os

OUTPUT_FILE = "/Users/mchevula/Desktop/PQBFL Latest/pqbfl_project adaptive ratcheting/Comprehensive_Report.tex"

def write_part_2():
    content = r"""
\section{Cryptographic Primitives}

\subsection{ML-KEM (Kyber)}
ML-KEM operates across three main functionalities bounded by the hardness of the Module Learning with Errors (M-LWE) problem: \texttt{KeyGen()}, \texttt{Encap()}, and \texttt{Decap()}. In PQBFL, Kyber-512 or Kyber-768 parameter sets establish the overarching post-quantum security bounds. 

\textit{Algorithm 1: Kyber Keypair Generation and Encapsulation}
\begin{algorithmic}[1]
\Function{KeyGen}{ }
    \State $A \gets R_{q}^{k \times k}$ \Comment{Public random matrix}
    \State $s, e \gets S_\eta$ \Comment{Secret and error vectors}
    \State $t \gets A \cdot s + e$
    \State \Return $pk \gets (A, t)$, $sk \gets s$
\EndFunction
\Function{Encap}{$pk$}
    \State $m \gets \{0,1\}^{256}$ \Comment{Random 256-bit seed}
    \State $K \gets KDF(m \| \mathcal{H}(pk))$ \Comment{Shared Secret}
    \State $(r, e_1, e_2) \gets PRF(m)$
    \State $u \gets A^T \cdot r + e_1$
    \State $v \gets t^T \cdot r + e_2 + \text{Encode}(m)$
    \State $c \gets (u, v)$
    \State \Return Ciphertext $c$, Shared Secret $K$
\EndFunction
\end{algorithmic}

The implementation strictly invokes the \texttt{pqcrypto} \textit{liboqs} backend compiled natively in C. This removes the variable-time branching flaws within previous iterations reliant upon \texttt{kyber-py}. 

\subsection{Constant-Time ECDH \& ECDSA}
Providing hybrid post-quantum agility, ECDH utilizing the SECP256k1 curve acts in tandem with Kyber. Both entities concurrently negotiate classical shared secrets alongside quantum ciphertexts. To avert 1-bit Nonce leaks (arising from arbitrary timing observations of scalar multiplications) that systematically crack ECDSA, the environment mandates OpenSSL backing, assuring $O(1)$ constant-time scalar derivation.

\subsection{AES-256-GCM AEAD}
Volatile models $\Delta w$ are encapsulated via Authenticated Encryption with Associated Data (AEAD) targeting the Advanced Encryption Standard at 256 bits, specifically inside Galois/Counter Mode (GCM). The mathematical construct strictly bonds ciphertexts against the operational phase and transmission vector:
\begin{equation}
C = \text{AES-GCM.Enc}(K_{i,j}, \eta, \Delta w_r, \mathcal{A})
\end{equation}
Here, the Associated Data is formulated strictly as $\mathcal{A} = \texttt{"pqbfl:"} \| dir \| \texttt{":"} \| r$. Tampering with the ciphertext alters the integrity tag, immediately terminating verification algorithms post-distribution. 

\subsection{Secure Execution Example: KDF \& Nonces}
\textbf{Solved Example - Nonce Injection:}
Instead of a predictable monotonic nonce sequence (e.g., extracting hash bits iteratively), AES-GCM derives a 96-bit randomized initialization vector (IV) per round strictly off $OS$ entropy pools (`os.urandom(12)`).
Let $r=1$:
\begin{enumerate}
    \item Sender retrieves random 12 bytes $\eta_{rand} = \text{0x4A8B...}$
    \item $C = \eta_{rand} \| \text{AES-GCM}(K_{1,1}, \eta_{rand}, \text{Model}, \mathcal{A})$
    \item Receiver unwraps $\eta_{rand} \gets C[0:12]$, $Cipher \gets C[12:]$.
\end{enumerate}
This universally eliminates deterministic IV-reuse attacks inherent in previous baseline applications evaluating multiple sessions consecutively.

\section{Proposed PQBFL Protocol}

The orchestrating protocol progresses in rigorous deterministic epochs categorized primarily into participant negotiation and subsequent iterative model federated distributions. Let index $j$ denote an asynchronous period initiated exclusively via a full Key Exchange mechanism (Asymmetric Ratchet) and $i$ indicate the synchronous iterative payload transmission index bounds (Symmetric Ratchet).

\subsection{Participant Key Publication}
At $t_0$, the Central Server creates immutable identity commitments. The server evaluates $kpk_b, ksk_b \gets \text{Kyber.KeyGen()}$ and $epk_b, esk_b \gets \text{ECDH.KeyGen()}$. The Server publishes the signature configuration $\mathcal{H}_{pk} = SHA256(kpk_b \| epk_b)$ strictly to the on-chain registry via Ethereum Smart Contracts. Participants retrieve these public bounds universally, anchoring physical verification processes inherently decentralized.

\subsection{Hybrid Session Establishment: Root Key Derivation}
Participant $A$ authenticates Server constraints and invokes Kyber encapsulation mechanisms rendering $c \gets Encap(kpk_b)$ and derives independent Kyber secret $SS_{k}$. Simultaneously $A$ determines $SS_{e}$ via Classical ECDH computations. $A$ commits local artifacts securely onto the chain framework and outputs ciphertext vectors $c$. The ensuing derivation of the mutual Root Key $RK_j$ ensures unconditional quantum-defensive equivalence:
\begin{align}
PRK_1 &= \text{HMAC-SHA256}(\sigma, SS_k) \label{eq:prk1}\\
PRK_2 &= \text{HMAC-SHA256}(PRK_1, SS_e) \label{eq:prk2}\\
RK_j &= \text{HKDF-Expand}(PRK_2, \texttt{"pqbfl:RK"}, 32)
\end{align}
If an adversarial entity mathematically circumvents secular ECDH cryptography (e.g., via Shor's Algorithm running on a 4000 logical qubit mainframe), the subsequent HMAC extraction maintains an identical defensive postulate established unilaterally by Kyber invariants. The session salt $\sigma \in \{0, 1\}^{256}$ is randomly drawn at each asymmetric reset.

\subsection{Iterative Off-Chain Model Transfer}
Post establishment, iterative execution epochs commence ($i = 0 \dots L_j-1$). Both endpoints sequentially iterate the symmetric parameters derived directly from $RK_j$:
\begin{align}
CK_0 &= \text{HMAC-SHA256}(RK_j, \texttt{"pqbfl:CK0"}) \\
CK_{i+1} &= \text{HMAC-SHA256}(CK_i, \texttt{"pqbfl:CK"}) \\
K_{i} &= \text{HMAC-SHA256}(CK_i, \texttt{"pqbfl:MK"}) \label{eq:sym_model}
\end{align}
Crucially, standard iterations permanently discard $CK_i$ promptly terminating evaluations. Consequently, evaluating iteration representations yields non-reversible transformations prohibiting adversaries deducing historic parameters. Each iteration encompasses the encryption of global models sent downstream alongside local models directed upstream sequentially utilizing $K_i$.
"""
    with open(OUTPUT_FILE, "a") as f:
        f.write(content)

if __name__ == "__main__":
    write_part_2()
    print("Part 2 written successfully.")
