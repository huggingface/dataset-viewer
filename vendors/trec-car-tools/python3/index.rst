trec-car-tools
==============

This is the documentation for ``trec-car-tools``, a Python 3 library for reading
and manipulating the `TREC Complex Answer Retrieval
<http://trec-car.cs.unh.edu/>`_ (CAR) dataset.

Getting started
---------------

This library requires Python 3.3 or greater. It can can be installed with
``setup.py`` ::

  python3 ./setup.py install

If you are using `Anaconda <https://www.anaconda.com/>`_, install the ``cbor``
library for Python 3.6: ::

  conda install -c laura-dietz cbor=1.0.0 
        
Once you have installed the library, you can download a `dataset
<http://trec-car.cs.unh.edu/datareleases/>`_ and start playing.

Reading the dataset
-------------------

The TREC CAR dataset consists of a number of different exports. These include,

 * Annotations files (also called "pages files") contain full Wikipedia pages and their contents
 * Paragraphs files contain only paragraphs disembodied from their pages
 * Outlines files contain only the section structure of pages and no textual content

To read an annotations file use the :func:`iter_annotations` function:

.. autofunction:: trec_car.read_data.iter_annotations

For instance, to list the page IDs of pages in a pages file one might write

.. code-block:: python

   for page in read_data.iter_annotations(open('train.test200.cbor', 'rb')):
       print(page.pageId)

Likewise, to read a paragraphs file the :func:`iter_paragraphs` function is
provided

.. autofunction:: trec_car.read_data.iter_paragraphs

To list the text of all paragraphs in a paragarphs file one might write,

.. code-block:: python

   for para in read_data.iter_paragraphs(open('train.test200.cbor', 'rb')):
       print(para.getText())

Basic types
-----------

.. class:: trec_car.read_data.PageName

   :class:`PageName` represents the natural language "name" of a page. Note that
   this means that it is not necessarily unique. If you need a unique handle for
   a page use :class:`PageId`.

.. class:: trec_car.read_data.PageId

   A :class:`PageId` is the unique identifier for a :class:`Page`.

The :class:`Page` type
----------------------

.. autoclass:: trec_car.read_data.Page
   :members:

.. autoclass:: trec_car.read_data.PageMetadata
   :members:

Types of pages
~~~~~~~~~~~~~~

.. autoclass:: trec_car.read_data.PageType

    The abstact base class.

.. autoclass:: trec_car.read_data.ArticlePage
.. autoclass:: trec_car.read_data.CategoryPage
.. autoclass:: trec_car.read_data.DisambiguationPage
.. autoclass:: trec_car.read_data.RedirectPage
   :members:

Page structure
--------------

The high-level structure of a :class:`Page` is captured by the subclasses of
:class:`PageSkeleton`.

.. autoclass:: trec_car.read_data.PageSkeleton
   :members:

.. autoclass:: trec_car.read_data.Para
   :members:
   :show-inheritance:

.. autoclass:: trec_car.read_data.Section
   :members:
   :show-inheritance:

.. autoclass:: trec_car.read_data.List
   :members:
   :show-inheritance:

.. autoclass:: trec_car.read_data.Image
   :members:
   :show-inheritance:

Paragraph contents
------------------

.. autoclass:: trec_car.read_data.Paragraph
   :members:

.. autoclass:: trec_car.read_data.ParaBody
   :members:

.. autoclass:: trec_car.read_data.ParaText
   :members:
   :show-inheritance:

.. autoclass:: trec_car.read_data.ParaLink
   :members:
   :show-inheritance:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

