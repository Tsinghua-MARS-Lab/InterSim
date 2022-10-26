# gulp-header-comment

[![Greenkeeper badge](https://badges.greenkeeper.io/mjeanroy/gulp-header-comment.svg)](https://greenkeeper.io/)
[![Build Status](https://travis-ci.org/mjeanroy/gulp-header-comment.svg?branch=master)](https://travis-ci.org/mjeanroy/gulp-header-comment)
[![Npm version](https://badge.fury.io/js/gulp-header-comment.svg)](https://badge.fury.io/js/gulp-header-comment)

gulp-header-comment is a [Gulp](https://github.com/gulpjs/gulp) extension to add comments to file(s) in the pipeline.

## Install

`npm install --save-dev gulp-header-comment`

## Usage

```javascript
const headerComment = require('gulp-header-comment');

gulp.src('**/*.js')
  .pipe(headerComment('License MIT'))
  .pipe(gulp.dest('./dist/'))
```

The generated comment will use:
- Block comment for JS, LESS, CSS and SASS files (i.e starts with `/**` and ends with `*/`).
- HTML comments for HTML files (i.e starts with `<!--` and ends with `-->`).
- Hash comments for `appcache` and hidden files (i.e starts with `#`).
- The default is the block comment otherwise.

See [https://github.com/3rd-Eden/commenting](commenting), as this package is used to generate comments according to file extensions.

## Templating

Header strings can use `lodash`, `moment` and data from `package.json`:

```javascript
const headerComment = require('gulp-header-comment');

gulp.src('**/*.js')
  .pipe(headerComment(`
    License: <%= pkg.license %>
    Generated on <%= moment().format('YYYY') %>
    Author: <%= _.capitalize(pkg.author) %>
  `))
  .pipe(gulp.dest('./dist/'))
```

You can also point to a file on disk:

```javascript
const headerComment = require('gulp-header-comment');

gulp.src('**/*.js')
  .pipe(headerComment({
    file: path.join(__dirname, 'header.txt'),
    encoding: 'utf-8', // Default is UTF-8
  }))
  .pipe(gulp.dest('./dist/'))
```

## ChangeLogs

See [here](https://github.com/mjeanroy/gulp-header-comment/blob/master/CHANGELOG.md).

## License

MIT License (MIT)

## Contributing

If you find a bug or think about enhancement, feel free to contribute and submit an issue or a pull request.
