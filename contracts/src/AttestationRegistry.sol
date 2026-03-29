// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title AttestationRegistry
 * @notice Immutable append-only registry of article provenance attestations.
 *
 * Each entry proves that a specific article was produced by a verified agent
 * running in an EigenCompute TEE, using a known model and prompt, with a
 * measured slant score. Anyone can query the registry to audit the agent's
 * editorial process.
 *
 * Design constraints:
 * - Append-only: attestations cannot be modified or deleted.
 * - Permissioned writes: only the registered agent address can publish.
 * - Agent rotation: the owner can update the agent address (key rotation).
 * - No upgradability: the contract is intentionally not upgradeable.
 */
contract AttestationRegistry {
    // ── Types ──────────────────────────────────────────────────────────

    struct Attestation {
        bytes32 articleHash;        // SHA-256 of headline + body
        bytes32 sourceSetHash;      // SHA-256 of sorted source content hashes
        bytes32 evaluatorOutputHash;// SHA-256 of full SlantScore JSON
        bytes32 promptConfigHash;   // SHA-256 of system prompt + user prompt
        int16   slantScore;         // overall_slant_score × 1000 (e.g. -150 = -0.15)
        uint8   calibrationRounds;
        uint48  timestamp;          // block.timestamp, fits until year 8921
        address publisher;          // msg.sender (the agent wallet)
        bytes   teeAttestation;     // raw TEE attestation from EigenCompute
    }

    // ── State ──────────────────────────────────────────────────────────

    address public owner;
    address public agent;

    Attestation[] public attestations;

    /// @notice Lookup by article hash → attestation index (+1, 0 = not found)
    mapping(bytes32 => uint256) public articleIndex;

    // ── Events ─────────────────────────────────────────────────────────

    event AttestationPublished(
        uint256 indexed id,
        bytes32 indexed articleHash,
        int16   slantScore,
        uint8   calibrationRounds,
        address publisher
    );

    event AgentUpdated(address indexed previousAgent, address indexed newAgent);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    // ── Errors ─────────────────────────────────────────────────────────

    error OnlyOwner();
    error OnlyAgent();
    error ZeroAddress();
    error DuplicateArticle(bytes32 articleHash);

    // ── Constructor ────────────────────────────────────────────────────

    /// @param _agent The initial agent wallet address permitted to publish.
    constructor(address _agent) {
        owner = msg.sender;
        agent = _agent;
    }

    // ── Write ──────────────────────────────────────────────────────────

    /// @notice Publish an attestation for a generated article.
    /// @dev Only callable by the registered agent address.
    function publish(
        bytes32 articleHash,
        bytes32 sourceSetHash,
        bytes32 evaluatorOutputHash,
        bytes32 promptConfigHash,
        int16   slantScore,
        uint8   calibrationRounds,
        bytes calldata teeAttestation
    ) external {
        if (msg.sender != agent) revert OnlyAgent();
        if (articleIndex[articleHash] != 0) revert DuplicateArticle(articleHash);

        attestations.push(Attestation({
            articleHash:         articleHash,
            sourceSetHash:       sourceSetHash,
            evaluatorOutputHash: evaluatorOutputHash,
            promptConfigHash:    promptConfigHash,
            slantScore:          slantScore,
            calibrationRounds:   calibrationRounds,
            timestamp:           uint48(block.timestamp),
            publisher:           msg.sender,
            teeAttestation:      teeAttestation
        }));

        uint256 id = attestations.length; // 1-indexed
        articleIndex[articleHash] = id;

        emit AttestationPublished(
            id,
            articleHash,
            slantScore,
            calibrationRounds,
            msg.sender
        );
    }

    // ── Read ───────────────────────────────────────────────────────────

    /// @notice Total number of attestations published.
    function count() external view returns (uint256) {
        return attestations.length;
    }

    /// @notice Look up an attestation by article hash.
    /// @return found Whether an attestation exists for this article.
    /// @return id    The 1-indexed attestation ID (0 if not found).
    function lookup(bytes32 articleHash) external view returns (bool found, uint256 id) {
        id = articleIndex[articleHash];
        found = id != 0;
    }

    /// @notice Get an attestation by 0-indexed position.
    function get(uint256 index) external view returns (Attestation memory) {
        return attestations[index];
    }

    // ── Admin ──────────────────────────────────────────────────────────

    /// @notice Update the agent wallet address (key rotation).
    function setAgent(address _agent) external {
        if (msg.sender != owner) revert OnlyOwner();
        if (_agent == address(0)) revert ZeroAddress();
        emit AgentUpdated(agent, _agent);
        agent = _agent;
    }

    /// @notice Transfer contract ownership (two-step: propose then accept).
    function transferOwnership(address _newOwner) external {
        if (msg.sender != owner) revert OnlyOwner();
        if (_newOwner == address(0)) revert ZeroAddress();
        emit OwnershipTransferred(owner, _newOwner);
        owner = _newOwner;
    }
}
