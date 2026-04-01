// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract PQBFL {
    uint256 public constant MIN_DEPOSIT_WEI = 0.01 ether;

    struct Project {
        bool exists;
        uint256 id;
        address server;
        uint256 nClients;
        uint256 clientCount;
        bytes32 hInitialModel;
        bytes32 hServerKeys; // h(kpk_b || epk_b)
        bool done;
        uint256 createdAt;
    }

    struct Client {
        bool exists;
        address addr;
        uint256 projectId;
        bytes32 hEpk;
        uint256 score;
        uint256 registeredAt;
    }

    struct Task {
        bool exists;
        uint256 round;
        uint256 taskId;
        uint256 projectId;
        address server;
        bytes32 hInf;   // h(Inf_b^r)
        bytes32 hPks;   // optional: hash of published keys for asymmetric ratchet
        uint256 deadline;
        uint256 publishedAt;
    }

    struct Update {
        bool exists;
        uint256 round;
        uint256 taskId;
        uint256 projectId;
        address client;
        bytes32 hInf;      // h(Inf_a^r)
        bytes32 hCtEpk;    // h(ct || epk_a) when asymmetric ratchet occurs, else 0x0
        uint256 submittedAt;
    }

    struct Feedback {
        bool exists;
        uint256 round;
        uint256 taskId;
        uint256 projectId;
        address client;
        bytes32 hUpdateInf;
        bytes32 hPks;
        int256 scoreDelta;
        bool terminate;
        uint256 time;
    }

    mapping(uint256 => Project) public projects;
    mapping(address => Client) public clients;
    mapping(uint256 => Task) public tasks;
    mapping(uint256 => mapping(address => Update)) public updatesByTaskAndClient;
    mapping(uint256 => mapping(address => Feedback)) public feedbackByTaskAndClient;

    event RegClient(address indexed cAddr, uint256 indexed id_p, uint256 sc, bytes32 h_epk);
    event RegProject(uint256 indexed id_p, uint256 nClients, address indexed sAddr, bytes32 h_M0, bytes32 h_pks);
    event TaskEvent(uint256 r, bytes32 h_Inf_b, bytes32 h_pks_r, uint256 indexed id_p, uint256 indexed id_t, uint256 nClients, uint256 D_t, uint256 time);
    event UpdateEvent(uint256 r, bytes32 h_Inf_a, bytes32 h_ct_epk, uint256 indexed id_p, uint256 indexed id_t, address indexed cAddr, uint256 time);
    event FeedbackEvent(uint256 r, uint256 indexed id_p, uint256 indexed id_t, bytes32 h_Inf_a, bytes32 h_pks_r, address indexed cAddr, int256 sc, bool T);
    event ProjectTerminate(uint256 r, uint256 indexed id_p, uint256 indexed id_t, uint256 time);

    modifier onlyServer(uint256 projectId) {
        require(projects[projectId].exists, "project missing");
        require(projects[projectId].server == msg.sender, "not server");
        _;
    }

    function registerProject(
        uint256 id_p,
        uint256 nClients,
        bytes32 h_M0,
        bytes32 h_pks
    ) external payable {
        require(!projects[id_p].exists, "project exists");
        require(nClients > 0, "nClients=0");
        require(msg.value >= MIN_DEPOSIT_WEI, "deposit too small");

        projects[id_p] = Project({
            exists: true,
            id: id_p,
            server: msg.sender,
            nClients: nClients,
            clientCount: 0,
            hInitialModel: h_M0,
            hServerKeys: h_pks,
            done: false,
            createdAt: block.timestamp
        });

        emit RegProject(id_p, nClients, msg.sender, h_M0, h_pks);
    }

    function registerClient(bytes32 h_epk, uint256 id_p) external {
        require(projects[id_p].exists, "project missing");
        require(!projects[id_p].done, "project done");
        require(projects[id_p].clientCount < projects[id_p].nClients, "project full");
        require(!clients[msg.sender].exists, "client already registered");

        clients[msg.sender] = Client({
            exists: true,
            addr: msg.sender,
            projectId: id_p,
            hEpk: h_epk,
            score: 0,
            registeredAt: block.timestamp
        });

        projects[id_p].clientCount += 1;

        emit RegClient(msg.sender, id_p, 0, h_epk);
    }

    function publishTask(
        uint256 r,
        bytes32 h_Inf_b,
        bytes32 h_pks_r,
        uint256 id_t,
        uint256 id_p,
        uint256 D_t
    ) external onlyServer(id_p) {
        require(!projects[id_p].done, "project done");

        tasks[id_t] = Task({
            exists: true,
            round: r,
            taskId: id_t,
            projectId: id_p,
            server: msg.sender,
            hInf: h_Inf_b,
            hPks: h_pks_r,
            deadline: D_t,
            publishedAt: block.timestamp
        });

        emit TaskEvent(r, h_Inf_b, h_pks_r, id_p, id_t, projects[id_p].nClients, D_t, block.timestamp);
    }

    function updateModel(
        uint256 r,
        bytes32 h_Inf_a,
        bytes32 h_ct_epk,
        uint256 id_t,
        uint256 id_p
    ) external {
        require(tasks[id_t].exists, "task missing");
        require(tasks[id_t].projectId == id_p, "wrong project");
        require(clients[msg.sender].exists, "client missing");
        require(clients[msg.sender].projectId == id_p, "client not in project");

        updatesByTaskAndClient[id_t][msg.sender] = Update({
            exists: true,
            round: r,
            taskId: id_t,
            projectId: id_p,
            client: msg.sender,
            hInf: h_Inf_a,
            hCtEpk: h_ct_epk,
            submittedAt: block.timestamp
        });

        emit UpdateEvent(r, h_Inf_a, h_ct_epk, id_p, id_t, msg.sender, block.timestamp);
    }

    function feedbackModel(
        uint256 r,
        uint256 id_t,
        uint256 id_p,
        address cAddr,
        bytes32 h_Inf_a,
        bytes32 h_pks_r,
        int256 sc,
        bool T
    ) external onlyServer(id_p) {
        require(tasks[id_t].exists, "task missing");
        require(!projects[id_p].done, "project done");

        feedbackByTaskAndClient[id_t][cAddr] = Feedback({
            exists: true,
            round: r,
            taskId: id_t,
            projectId: id_p,
            client: cAddr,
            hUpdateInf: h_Inf_a,
            hPks: h_pks_r,
            scoreDelta: sc,
            terminate: T,
            time: block.timestamp
        });

        _updateScore(cAddr, sc);

        emit FeedbackEvent(r, id_p, id_t, h_Inf_a, h_pks_r, cAddr, sc, T);

        if (T) {
            projects[id_p].done = true;
            emit ProjectTerminate(r, id_p, id_t, block.timestamp);
        }
    }

    function finishProject(uint256 id_p, uint256 r, uint256 id_t) external onlyServer(id_p) {
        require(!projects[id_p].done, "already done");
        projects[id_p].done = true;
        emit ProjectTerminate(r, id_p, id_t, block.timestamp);
    }

    function _updateScore(address cAddr, int256 sc) internal {
        if (!clients[cAddr].exists) return;

        if (sc < 0) {
            uint256 delta = uint256(-sc);
            if (delta >= clients[cAddr].score) {
                clients[cAddr].score = 0;
            } else {
                clients[cAddr].score -= delta;
            }
        } else {
            clients[cAddr].score += uint256(sc);
        }
    }
}
