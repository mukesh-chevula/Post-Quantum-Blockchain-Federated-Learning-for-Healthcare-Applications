async function main() {
  const PQBFL = await ethers.getContractFactory("PQBFL");
  const pqbfl = await PQBFL.deploy();
  await pqbfl.waitForDeployment();
  const address = await pqbfl.getAddress();
  console.log("PQBFL deployed to:", address);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
