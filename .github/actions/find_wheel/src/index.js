const core = require('@actions/core');
const glob = require('glob');
const path = require('path');

async function getPythonVersion() {
  const { stdout } = await execAsync('python --version');
  const versionMatch = stdout.match(/Python (\d+)\.(\d+)\.(\d+)/);
  if (versionMatch) {
    return {
      major: versionMatch[1],
      minor: versionMatch[2],
      patch: versionMatch[3]
    };
  } else {
    throw new Error('Unable to detect Python version');
  }
}

async function run() {
  try {
    const localWheelDir = core.getInput('wheels_dir');
    const packageName = core.getInput('package_name');

    const pythonVersion = await getPythonVersion();
    core.debug(`Detected Python version: ${JSON.stringify(pythonVersion)}`);
        
    const wheelsFound = [];
    if (localWheelDir) {
      const pattern = `${packageName}*.whl`;
      const globber = await glob.create(path.posix.join(localWheelDir, pattern));
      const wheels = await globber.glob();      
      core.debug(`Found wheels: ${wheels}`);

      for (const whl of wheels) {
        const wheelName = path.basename(whl).split('-')[0];
        const wheelPythonVersion = wheelName.match(/cp(\d{2,3})/);
        if (
          !wheelPythonVersion ||
          wheelPythonVersion[1] === `${pythonVersion.major}${pythonVersion.minor}`
        ) {
          wheelsFound.push(whl);
        }
      }
    }
    core.debug(`Resolved local wheels: ${JSON.stringify(wheelsFound)}`);

    if (wheelsFound.length === 0) {
      core.setFailed(`No files found matching pattern "${pattern}"`);
      return;
    } else if (wheelsFound.length > 1) {
      core.setFailed(`Multiple files found matching pattern "${pattern}"`);
      return;
    } else {
      core.info(`Found ${wheelsFound[0]} matching pattern "${pattern}"`);
    }

    core.setOutput('wheel_path', wheelsFound[0]);
  } catch (error) {
    core.setFailed(error.message);
  }
}

run();
