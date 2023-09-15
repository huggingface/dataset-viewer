# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.
# adapted from https://docs.mongoengine.org/guide/migration.html#post-processing-checks

import logging
from collections.abc import Callable, Iterator
from typing import Optional, TypeVar

from mongoengine import Document
from pymongo.collection import Collection

# --- some typing subtleties, see https://github.com/sbdchd/mongo-types
U = TypeVar("U", bound=Document)
DocumentClass = type[U]
CustomValidation = Callable[[U], None]
# --- end


def get_random_oids(collection: Collection, sample_size: int) -> list[int]:
    pipeline = [{"$project": {"_id": 1}}, {"$sample": {"size": sample_size}}]
    return [s["_id"] for s in collection.aggregate(pipeline)]


def get_random_documents(DocCls: DocumentClass[Document], sample_size: int) -> Iterator[Document]:
    doc_collection = DocCls._get_collection()
    random_oids = get_random_oids(doc_collection, sample_size)
    return DocCls.objects(pk__in=random_oids)  # type: ignore


def check_documents(
    DocCls: DocumentClass[Document],
    sample_size: int,
    custom_validation: Optional[CustomValidation[Document]] = None,
) -> None:
    for doc in get_random_documents(DocCls, sample_size):
        try:
            # general validation (types and values)
            doc.validate()

            # load all subfields,
            # this may trigger additional queries if you have ReferenceFields
            # so it may be slow
            for field in doc._fields:
                try:
                    getattr(doc, field)
                except Exception:
                    logging.error(f"Could not load field {field} in Document {doc.pk}. Document: {doc.to_json()}")
                    raise

            # custom validation
            if custom_validation is not None:
                custom_validation(doc)
        except Exception as e:
            logging.error(f"Validation error on document {doc.pk}: {e}. Document: {doc.to_json()}")
            raise e
