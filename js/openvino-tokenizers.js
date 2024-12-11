const os = require('node:os');
const path = require('node:path');

module.exports = {
  path: getPathToBinary({
    platform: os.platform(),
    arch: os.arch(),
  }),
  getPathToBinary,
};

function getPathToBinary(osProps) {
  const { arch, platform } = osProps;

  if (platform === 'win32' && arch !== 'x64')
    throw new Error(`Version for windows and '${arch}' is not supported.`);

  return path.join(
    __dirname,
    'bin/runtime',
    libOrBin(platform),
    getDirnameByArch(arch),
    platform === 'linux' ? '' : 'Release',
    getBinaryFilename(platform),
  );
}

function getDirnameByArch(arch) {
  switch (arch) {
    case 'x64':
      return 'intel64';
    case 'arm64':
    case 'armhf':
      return 'arm64';
    default:
      throw new Error(`Unsupported architecture: ${arch}`);
  }
}

function libOrBin(platform) {
  switch (platform) {
    case 'win32':
      return 'bin';
    default:
      return 'lib';
  }
}

function getBinaryFilename(platform) {
  switch (platform) {
    case 'win32':
      return 'openvino_tokenizers.dll';
    case 'linux':
      return 'libopenvino_tokenizers.so';
    case 'darwin':
      return 'libopenvino_tokenizers.dylib';
    default:
      throw new Error(`Unsupported platform: ${platform}`);
  }
}
