const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying with account:", deployer.address);

  // The agent address that will be permitted to publish attestations.
  // In production this comes from AgentKit; for local dev, use the deployer.
  const agentAddress =
    process.env.AGENT_WALLET_ADDRESS || deployer.address;

  const Registry = await hre.ethers.getContractFactory("AttestationRegistry");
  const registry = await Registry.deploy(agentAddress);
  await registry.waitForDeployment();

  const address = await registry.getAddress();
  console.log("AttestationRegistry deployed to:", address);
  console.log("Agent address:", agentAddress);
  console.log(
    "\nSet ATTESTATION_CONTRACT_ADDRESS=%s in your .env",
    address
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
