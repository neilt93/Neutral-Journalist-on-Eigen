const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("AttestationRegistry", function () {
  let registry, owner, agent, other;

  // Deterministic test hashes
  const articleHash = ethers.id("article-content-hash");
  const sourceSetHash = ethers.id("source-set-hash");
  const evaluatorHash = ethers.id("evaluator-output-hash");
  const promptHash = ethers.id("prompt-config-hash");
  const slantScore = -120; // -0.12
  const calibrationRounds = 2;
  const teeAttestation = ethers.toUtf8Bytes("tee-attestation-data");

  beforeEach(async function () {
    [owner, agent, other] = await ethers.getSigners();
    const Registry = await ethers.getContractFactory("AttestationRegistry");
    registry = await Registry.deploy(agent.address);
    await registry.waitForDeployment();
  });

  describe("Deployment", function () {
    it("sets owner and agent correctly", async function () {
      expect(await registry.owner()).to.equal(owner.address);
      expect(await registry.agent()).to.equal(agent.address);
    });

    it("starts with zero attestations", async function () {
      expect(await registry.count()).to.equal(0);
    });
  });

  describe("Publishing", function () {
    it("allows the agent to publish an attestation", async function () {
      const tx = await registry
        .connect(agent)
        .publish(
          articleHash,
          sourceSetHash,
          evaluatorHash,
          promptHash,
          slantScore,
          calibrationRounds,
          teeAttestation
        );

      await expect(tx)
        .to.emit(registry, "AttestationPublished")
        .withArgs(1, articleHash, slantScore, calibrationRounds, agent.address);

      expect(await registry.count()).to.equal(1);
    });

    it("stores attestation data correctly", async function () {
      await registry
        .connect(agent)
        .publish(
          articleHash,
          sourceSetHash,
          evaluatorHash,
          promptHash,
          slantScore,
          calibrationRounds,
          teeAttestation
        );

      const att = await registry.get(0);
      expect(att.articleHash).to.equal(articleHash);
      expect(att.sourceSetHash).to.equal(sourceSetHash);
      expect(att.evaluatorOutputHash).to.equal(evaluatorHash);
      expect(att.promptConfigHash).to.equal(promptHash);
      expect(att.slantScore).to.equal(slantScore);
      expect(att.calibrationRounds).to.equal(calibrationRounds);
      expect(att.publisher).to.equal(agent.address);
    });

    it("rejects non-agent callers", async function () {
      await expect(
        registry
          .connect(other)
          .publish(
            articleHash,
            sourceSetHash,
            evaluatorHash,
            promptHash,
            slantScore,
            calibrationRounds,
            teeAttestation
          )
      ).to.be.revertedWithCustomError(registry, "OnlyAgent");
    });

    it("rejects duplicate article hashes", async function () {
      await registry
        .connect(agent)
        .publish(
          articleHash,
          sourceSetHash,
          evaluatorHash,
          promptHash,
          slantScore,
          calibrationRounds,
          teeAttestation
        );

      await expect(
        registry
          .connect(agent)
          .publish(
            articleHash,
            sourceSetHash,
            evaluatorHash,
            promptHash,
            slantScore,
            calibrationRounds,
            teeAttestation
          )
      ).to.be.revertedWithCustomError(registry, "DuplicateArticle");
    });
  });

  describe("Lookup", function () {
    it("returns found=true for published articles", async function () {
      await registry
        .connect(agent)
        .publish(
          articleHash,
          sourceSetHash,
          evaluatorHash,
          promptHash,
          slantScore,
          calibrationRounds,
          teeAttestation
        );

      const [found, id] = await registry.lookup(articleHash);
      expect(found).to.be.true;
      expect(id).to.equal(1);
    });

    it("returns found=false for unknown articles", async function () {
      const [found, id] = await registry.lookup(ethers.id("unknown"));
      expect(found).to.be.false;
      expect(id).to.equal(0);
    });
  });

  describe("Multiple attestations", function () {
    it("handles sequential publishes with correct IDs", async function () {
      const hash1 = ethers.id("article-1");
      const hash2 = ethers.id("article-2");
      const hash3 = ethers.id("article-3");

      await registry
        .connect(agent)
        .publish(hash1, sourceSetHash, evaluatorHash, promptHash, -50, 1, teeAttestation);
      await registry
        .connect(agent)
        .publish(hash2, sourceSetHash, evaluatorHash, promptHash, 0, 1, teeAttestation);
      await registry
        .connect(agent)
        .publish(hash3, sourceSetHash, evaluatorHash, promptHash, 100, 3, teeAttestation);

      expect(await registry.count()).to.equal(3);

      const [, id1] = await registry.lookup(hash1);
      const [, id2] = await registry.lookup(hash2);
      const [, id3] = await registry.lookup(hash3);
      expect(id1).to.equal(1);
      expect(id2).to.equal(2);
      expect(id3).to.equal(3);
    });
  });

  describe("Admin", function () {
    it("allows owner to rotate agent", async function () {
      await expect(registry.setAgent(other.address))
        .to.emit(registry, "AgentUpdated")
        .withArgs(agent.address, other.address);

      expect(await registry.agent()).to.equal(other.address);

      // Old agent can no longer publish
      await expect(
        registry
          .connect(agent)
          .publish(
            articleHash,
            sourceSetHash,
            evaluatorHash,
            promptHash,
            slantScore,
            calibrationRounds,
            teeAttestation
          )
      ).to.be.revertedWithCustomError(registry, "OnlyAgent");

      // New agent can publish
      await registry
        .connect(other)
        .publish(
          articleHash,
          sourceSetHash,
          evaluatorHash,
          promptHash,
          slantScore,
          calibrationRounds,
          teeAttestation
        );
      expect(await registry.count()).to.equal(1);
    });

    it("rejects non-owner agent rotation", async function () {
      await expect(
        registry.connect(other).setAgent(other.address)
      ).to.be.revertedWithCustomError(registry, "OnlyOwner");
    });
  });
});
