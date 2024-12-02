const assert = require('node:assert/strict');
const { test, describe } = require('node:test');

const { getPathToBinary } = require('../openvino-tokenizers');

const parentDir = __dirname.split('/').slice(0, -1).join('/');

describe('getPathToBinary for x64', () => {
  test('should map win32 to windows and x64 to intel64', () => {
    const path = getPathToBinary({ platform: 'win32', arch: 'x64' });
    const [, relatedFromBin] = path.split(parentDir);

    assert.equal(
      relatedFromBin,
      '/bin/runtime/bin/intel64/Release/openvino_tokenizers.dll',
    );
  });

  test('should map linux to linux and x64 to intel64', () => {
    const path = getPathToBinary({ platform: 'linux', arch: 'x64' });
    const [, relatedFromBin] = path.split(parentDir);

    assert.equal(
      relatedFromBin,
      '/bin/runtime/lib/intel64/libopenvino_tokenizers.so',
    );
  });

  test('should map darwin to macos and x64 to intel64', () => {
    const path = getPathToBinary({ platform: 'darwin', arch: 'x64' });
    const [, relatedFromBin] = path.split(parentDir);

    assert.equal(
      relatedFromBin,
      '/bin/runtime/lib/intel64/Release/libopenvino_tokenizers.dylib',
    );
  });
});

describe('getPathToBinary for arm64', () => {
  test('should map win32 to windows and arm64 to arm64', () => {
    assert.throws(
      () => getPathToBinary({ platform: 'win32', arch: 'arm64' }),
      new Error(`Version for windows and 'arm64' is not supported.`),
    );
  });

  test('should map linux to linux and arm64 to arm64', () => {
    const path = getPathToBinary({ platform: 'linux', arch: 'arm64' });
    const [, relatedFromBin] = path.split(parentDir);

    assert.equal(
      relatedFromBin,
      '/bin/runtime/lib/arm64/libopenvino_tokenizers.so',
    );
  });

  test('should map darwin to macos and arm64 to arm64', () => {
    const path = getPathToBinary({ platform: 'darwin', arch: 'arm64' });
    const [, relatedFromBin] = path.split(parentDir);

    assert.equal(
      relatedFromBin,
      '/bin/runtime/lib/arm64/Release/libopenvino_tokenizers.dylib',
    );
  });
});

describe('getPathToBinary for armhf', () => {
  test('should map win32 to windows and armhf to arm64', () => {
    assert.throws(
      () => getPathToBinary({ platform: 'win32', arch: 'armhf' }),
      new Error(`Version for windows and 'armhf' is not supported.`),
    );
  });

  test('should map linux to linux and armhf to arm64', () => {
    const path = getPathToBinary({ platform: 'linux', arch: 'armhf' });
    const [, relatedFromBin] = path.split(parentDir);

    assert.equal(
      relatedFromBin,
      '/bin/runtime/lib/arm64/libopenvino_tokenizers.so',
    );
  });

  test('should map darwin to macos and armhf to arm64', () => {
    const path = getPathToBinary({ platform: 'darwin', arch: 'armhf' });
    const [, relatedFromBin] = path.split(parentDir);

    assert.equal(
      relatedFromBin,
      '/bin/runtime/lib/arm64/Release/libopenvino_tokenizers.dylib',
    );
  });
});

describe('getPathToBinary for unsupported platform', () => {
  test('should throw an error for unsupported platform', () => {
    assert.throws(() => getPathToBinary({ platform: 'unsupported', arch: 'x64' }));
  });
});

