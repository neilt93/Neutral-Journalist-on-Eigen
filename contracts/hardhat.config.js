require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config({ path: "../.env" });

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: "0.8.24",
  paths: {
    sources: "./src",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts",
  },
  networks: {
    // Local dev
    hardhat: {},
    // Testnet deployment
    sepolia: {
      url: process.env.SEPOLIA_RPC_URL || "",
      accounts: process.env.AGENT_WALLET_PRIVATE_KEY
        ? [process.env.AGENT_WALLET_PRIVATE_KEY]
        : [],
    },
  },
};
