/**
 * Path to the tokenizer extension library for the current systeam
 */
export const path: string;
/**
 * Returns the path to the tokenizer extension library based on the specified architecture and platform.
 *
 * @param {Object} args - The system information.
 * @param {string} args.arch - The CPU architecture (e.g., 'x64', 'arm64').
 * @param {string} args.platform - The operating system platform (e.g., 'win32', 'linux', 'darwin').
 * @returns {string} The path to the tokenizer extension library.
 */
export function getPathToBinary(args: {arch: string, platform: string}): string;
