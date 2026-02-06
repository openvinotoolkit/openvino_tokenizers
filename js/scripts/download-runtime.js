
const os = require('node:os');
const AdmZip = require('adm-zip');
const { join } = require('node:path');
const BinaryManager = require('openvino-node/scripts/lib/binary-manager');

const packageJson = require('../package.json');

class TokenizersBinaryManager extends BinaryManager {
  getVersion() {
    return `${ this.binaryConfig.version || this.version }.0`;
  }

  getPlatformLabel() {
    return this.convertPlatformLabel(os.platform());
  }

  getArchLabel() {
    return this.convertArchLabel(os.arch());
  }

  getExtension() {
    return os.platform() === 'win32' ? 'zip' : 'tar.gz';
  }

  convertPlatformLabel(platform) {
    switch(platform) {
      case 'win32':
        return 'windows';
      case 'linux':
        return 'rhel8';
      case 'darwin':
        return 'macos_12_6';
    }
  }

  convertArchLabel(arch) {
    switch(arch) {
      case 'x64':
        return 'x86_64';
      case 'arm64':
      case 'armhf':
        return 'arm64';
    }
  }

  /**
   * Unarchive tar, tar.gz or zip archives.
   *
   * @function unarchive
   * @param {string} archivePath - Path to archive.
   * @param {string} dest - Path where to unpack.
   * @returns {Promise<void>}
   */
  static unarchive(archivePath, dest) {
    if (archivePath.endsWith('.zip')) {
      const zip = new AdmZip(archivePath);

      return new Promise((resolve, reject) => {
        zip.extractAllToAsync(dest, true, (err) => {
          if (err) {
            reject(err);
          } else {
            resolve();
          }
        });
      });
    }

    return BinaryManager.unarchive(archivePath, dest);
  }
}

if (require.main === module) main();

async function main() {
  if (!TokenizersBinaryManager.isCompatible()) process.exit(1);

  const force = process.argv.includes('-f');
  const ignoreIfExists = process.argv.includes('--ignore-if-exists');

  const { env } = process;
  const proxy = env.http_proxy || env.HTTP_PROXY || env.npm_config_proxy;

  await TokenizersBinaryManager.prepareBinary(
    join(__dirname, '..'),
    packageJson.binary.version || packageJson.version,
    packageJson.binary,
    { force, ignoreIfExists, proxy },
  );
}
