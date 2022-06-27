<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Generating the documentation

To generate the documentation, you first have to build it. Several packages are necessary to build the doc,
you can install them with the following command, in this directory:

```bash
make install
```

---

**NOTE**

You only need to generate the documentation to inspect it locally (if you're planning changes and want to
check how they look like before committing for instance). You don't have to commit the built documentation.

---

## Building the documentation

Once you have setup the `doc-builder` and additional packages, you can generate the documentation by typing the
following command:

```bash
BUILD_DIR=/tmp/doc-datasets-server/ make build
```

You can adapt the `BUILD_DIR` environment variable to set any temporary folder that you prefer. This command will create it and generate
the MDX files that will be rendered as the documentation on the main website. You can inspect them in your favorite
Markdown editor.

---

**NOTE**

It's not possible to see locally how the final documentation will look like for now. Once you have opened a PR, you
will see a bot add a comment to a link where the documentation with your changes lives.

---

## Adding a new element to the navigation bar

Accepted files are Markdown (.md or .mdx).

Create a file with its extension and put it in the source directory. You can then link it to the toc-tree by putting
the filename without the extension in the [`_toctree.yml`](https://github.com/huggingface/datasets-server/blob/main/docs/source/_toctree.yml) file.

## Adding an image

Due to the rapidly growing repository, it is important to make sure that no files that would significantly weigh down the repository are added. This includes images, videos and other non-text files. We prefer to leverage a hf.co hosted `dataset` like
the ones hosted on [`hf-internal-testing`](https://huggingface.co/hf-internal-testing) in which to place these files and reference
them by URL. We recommend putting them in the following dataset: [huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images).
If an external contribution, feel free to add the images to your PR and ask a Hugging Face member to migrate your images
to this dataset.
