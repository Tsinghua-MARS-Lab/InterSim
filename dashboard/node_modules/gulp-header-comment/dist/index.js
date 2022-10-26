/**
 * The MIT License (MIT)
 *
 * Copyright (c) 2017-2020 Mickael Jeanroy
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
'use strict';

const fs = require('fs');

const path = require('path');

const stringDecoder = require('string_decoder');

const log = require('fancy-log');

const colors = require('ansi-colors');

const PluginError = require('plugin-error');

const _ = require('lodash');

const moment = require('moment');

const commenting = require('commenting');

const through = require('through2');

const applySourceMap = require('vinyl-sourcemaps-apply');

const MagicString = require('magic-string');

const Q = require('q');

module.exports = function gulpHeaderComment(options = {}) {
  const separator = _.isObject(options) && _.isString(options.separator) ? options.separator : '\n';
  const cwd = options.cwd || process.cwd();
  const pkgPath = path.join(cwd, 'package.json');
  const pkg = fs.existsSync(pkgPath) ? require(pkgPath) : {};
  return through.obj((file, encoding, cb) => {
    if (file.isNull() || file.isDirectory()) {
      return cb(null, file);
    }

    const filePath = file.path;
    read(options, encoding).then(content => {
      const extension = getExtension(file);
      const type = extension.slice(1);
      const header = generateHeader(content, extension, pkg, createFileObject(cwd, filePath));

      if (file.isBuffer()) {
        updateFileContent(file, type, header, separator);
      } else if (file.isStream()) {
        pipeFileContent(file, type, header, separator);
      }

      cb(null, file);
    }).catch(err => {
      // Log error.
      log.error(colors.red(`gulp-header-comment: ${err}`)); // Wrap error.

      cb(new PluginError('gulp-header-comment', err));
    });
  });
};
/**
 * Add header to given file content.
 *
 * @param {Object} file Original file.
 * @param {string} type File type.
 * @param {string} header Header to add to given file.
 * @param {string} separator The separator to use between header and file content.
 * @return {void}
 */


function updateFileContent(file, type, header, separator) {
  const input = file.contents.toString();
  const fname = file.relative;
  const result = transform(fname, input, type, header, separator);
  file.contents = toBuffer(result.code);

  if (file.sourceMap && result.map) {
    applySourceMap(file, result.map);
  }
}
/**
 * Create file descriptor object.
 *
 * @param {string} cwd Working directory.
 * @param {string} filePath File absolute path.
 * @return {Object} An object containing file information.
 */


function createFileObject(cwd, filePath) {
  if (!filePath) {
    return null;
  }

  const normalizedCwd = path.normalize(cwd);
  const normalizedPath = path.normalize(filePath);
  const filename = path.basename(normalizedPath);
  const dirname = path.dirname(normalizedPath);
  return {
    name: filename,
    dir: dirname,
    path: normalizedPath,
    relativePath: path.normalize(path.relative(normalizedCwd, normalizedPath)),
    relativeDir: path.normalize(path.relative(normalizedCwd, dirname))
  };
}
/**
 * Stream new file content.
 *
 * @param {File} file Given input file.
 * @param {string} type The file type.
 * @param {string} header Header to add to given file.
 * @param {string} separator Separator between header and file content.
 * @return {void}
 */


function pipeFileContent(file, type, header, separator) {
  return maySkipFirstLine(type) ? transformFileStreamContent(file, type, header, separator) : prependPipeStream(file, header, separator);
}
/**
 * A very simple function that prepend header to file stream input.
 *
 * @param {File} file The original file stream.
 * @param {string} header The header to prepend to original file stream.
 * @param {string} separator The separator to use between header and file content.
 * @return {void}
 */


function prependPipeStream(file, header, separator) {
  const stream = through();
  stream.write(toBuffer(header + separator));
  file.contents = file.contents.pipe(stream);
}
/**
 * A very simple function that prepend header to file stream input.
 *
 * @param {File} file The original file stream.
 * @param {string} type File type.
 * @param {string} header The header to prepend to original file stream.
 * @param {string} separator The separator to use between header and file content.
 * @return {void}
 */


function transformFileStreamContent(file, type, header, separator) {
  file.contents = file.contents.pipe(through(function transformFunction(chunk, enc, cb) {
    const decoder = new stringDecoder.StringDecoder();
    const rawChunk = decoder.end(chunk);
    const fname = file.relative;
    const result = transform(fname, rawChunk, type, header, separator);
    const newContent = result.code;

    if (file.sourceMap && result.map) {
      applySourceMap(file, result.map);
    } // eslint-disable-next-line no-invalid-this


    this.push(newContent);
    cb();
  }));
}
/**
 * Get extension for given file.
 *
 * @param {Object} file The file.
 * @return {string} File extension.
 */


function getExtension(file) {
  const ext = path.extname(file.path);
  return ext ? ext.toLowerCase() : ext;
}
/**
 * Generate header from given template.
 *
 * @param {string} content Template of header.
 * @param {string} extension Target file extension.
 * @param {Object} pkg The `package.json` descriptor that will be injected when template will be evaluated.
 * @param {Object} file The `file` being processed (contains `path`, `name` and `dir` entries).
 * @return {string} Interpolated header.
 */


function generateHeader(content, extension, pkg, file) {
  const templateFn = _.template(content);

  const template = templateFn({
    _,
    moment,
    pkg,
    file
  });
  return commenting(template.trim(), {
    extension
  });
}
/**
 * Add header to given file content.
 *
 * @param {string} fname File name.
 * @param {string} content Original file content.
 * @param {string} type Original file type.
 * @param {string} header The header to add.
 * @param {string} separator The separator to use between original file content and separator.
 * @return {string} The resulting file content.
 */


function transform(fname, content, type, header, separator) {
  const magicStr = new MagicString(content);

  if (!maySkipFirstLine(type)) {
    prependHeader(magicStr, header, separator);
  } else {
    const lineSeparator = '\n';
    const lines = content.split(lineSeparator);
    const firstLine = lines[0].toLowerCase();

    const trimmedFirstLine = _.trim(firstLine);

    if (!shouldSkipFirstLine(type, trimmedFirstLine)) {
      prependHeader(magicStr, header, separator);
    } else {
      magicStr.appendRight(lines[0].length, separator + separator + header);
    }
  }

  return {
    code: magicStr.toString(),
    map: magicStr.generateMap({
      file: `${fname}.map`,
      source: fname,
      includeContent: true,
      hires: true
    })
  };
}
/**
 * Prepend header to given file content.
 *
 * @param {MagicString} magicStr Original file content.
 * @param {string} header Header to prepend.
 * @param {string} separator The separator between header and file content.
 */


function prependHeader(magicStr, header, separator) {
  magicStr.prepend(header + separator);
} // Set of checker function for each file type that may start with a prolog ling.


const prologCheckers = {
  /**
   * Check that given line is the `DOCTYPE` line.
   *
   * @param {string} line The line to check.
   * @return {boolean} `true` if given is an HTML `DOCTYPE`, `false` otherwise.
   */
  htm(line) {
    return this.html(line);
  },

  /**
   * Check that given line is the `DOCTYPE` line.
   *
   * @param {string} line The line to check.
   * @return {boolean} `true` if given is an HTML `DOCTYPE`, `false` otherwise.
   */
  html(line) {
    return _.startsWith(line, '<!doctype');
  },

  /**
   * Check that given line is the `XML` line.
   *
   * @param {string} line The line to check.
   * @return {boolean} `true` if given is an HTML `XML`, `false` otherwise.
   */
  xml(line) {
    return _.startsWith(line, '<?xml');
  },

  /**
   * Check that given line is the `XML` line.
   *
   * @param {string} line The line to check.
   * @return {boolean} `true` if given is an HTML `XML`, `false` otherwise.
   */
  svg(line) {
    return this.xml(line);
  },

  /**
   * Check that given line is the `JS` line.
   *
   * @param {string} line The line to check.
   * @return {boolean} `true` if given is an JS `#!`, `false` otherwise.
   */
  js(line) {
    return _.startsWith(line, '#!');
  }

};
/**
 * Check if given file type (`js`, `xml`, etc.) may start with a given prolog (a.k.a declaration).
 * For example, XML/SVG files may include a prolog such as `<?xml version="1.0" encoding="UTF-8"?>` and this
 * prolog must always be the first line (before anything else, including comments).
 *
 * @param {string} type File type.
 * @return {boolean} `true` if given file type may start with a prolog, `false` otherwise.
 */

function maySkipFirstLine(type) {
  return _.has(prologCheckers, type);
}
/**
 * Check if given line should be skipped before adding header content.
 *
 * @param {string} type File type.
 * @param {string} line The first line of given file.
 * @return {boolean} `true` if given line should be skipped, `false` otherwise.
 */


function shouldSkipFirstLine(type, line) {
  return prologCheckers[type](line);
}
/**
 * Read file specified by given options.
 *
 * @param {Object} options Read options.
 * @param {string} defaultEncoding The default encoding to use.
 * @return {Promise<string>} A promise resolved with file content.
 */


function read(options, defaultEncoding) {
  if (_.isString(options)) {
    return Q.when(options);
  }

  const file = options.file;
  const encoding = options.encoding || defaultEncoding || 'utf-8';
  const deferred = Q.defer();
  fs.readFile(file, {
    encoding
  }, (err, data) => {
    return err ? deferred.reject(err) : deferred.resolve(data);
  });
  return deferred.promise;
}
/**
 * Creates a new Buffer containing string.
 *
 * @param {string} rawString The string content.
 * @return {Buffer} Node Buffer.
 */


function toBuffer(rawString) {
  return Buffer.from ? Buffer.from(rawString) : new Buffer(rawString);
}