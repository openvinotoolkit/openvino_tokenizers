# OpenVINO Tokenizers

This package contains binaries extension for
[openvino-node](https://www.npmjs.com/package/openvino-node) package

## Installation

```bash
npm install openvino-tokenizers-node
```
After package installation script that downloads and extracts binaries will be executed. Binaries will be placed in the `node_modules/openvino-tokenizers-node/bin` directory.

This package is a part of [openvino-node](https://www.npmjs.com/package/openvino-node) package and should be used with it. Version of this package should be the same as version of openvino-node package.

## Usage

```javascript
const { addon: ov } = require('openvino-node');
const openvinoTokenizers = require('openvino-tokenizers-node');

const core = new ov.Core();
core.addExtension(openvinoTokenizers.path); // Add tokenizers extension

// Load tokenizers model
// ...
```

[License](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE)

Copyright © 2018-2025 Intel Corporation
