import tempfile
import unittest
from pathlib import Path
from typing import final

from scripts.data.openimages import OpenImagesSource, license_name

CC_BY = "https://creativecommons.org/licenses/by/2.0/"
FILES = {
    "oidv7-class-descriptions-boxable.csv": (
        "LabelName,DisplayName\n/m/dog,Dog\n/m/man,Man\n/m/horse,Horse\n"
        "/m/cattle,Cattle\n/m/orange,Orange (fruit)\n"
    ),
    "oidv6-attributes-description.csv": "/m/run,Run\n",
    "validation-images-with-rotation.csv": (
        "ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,"
        "Author,Title,OriginalSize,OriginalMD5,Thumbnail300KURL,Rotation\n"
        f"a,validation,https://flickr/a.jpg,https://flickr/a,{CC_BY},,Ann,A,1,x,,0.0\n"
        f"b,validation,https://flickr/b.jpg,https://flickr/b,{CC_BY},,Bob,B,1,x,,90.0\n"
        f"c,validation,https://flickr/c.jpg,https://flickr/c,{CC_BY},,Cy,C,1,x,,\n"
        f"d,validation,https://flickr/d.jpg,https://flickr/d,{CC_BY},,Di,D,1,x,,0.0\n"
    ),
    "validation-annotations-bbox.csv": (
        "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,"
        "IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n"
        "a,xclick,/m/dog,1,0.1,0.6,0.2,0.9,0,0,0,0,0\n"  # kept
        "a,xclick,/m/dog,1,0.0,0.5,0.0,0.5,0,0,0,0,0\n"  # second dog in a: one per image
        "b,xclick,/m/dog,1,0.1,0.6,0.2,0.9,0,0,0,0,0\n"  # rotated image
        "c,xclick,/m/dog,1,0.1,0.6,0.2,0.9,0,0,0,0,0\n"  # rotation unknown
        "d,xclick,/m/dog,1,0.1,0.6,0.2,0.9,0,0,1,0,0\n"  # group of dogs
        "d,xclick,/m/cattle,1,0.1,0.15,0.2,0.9,0,0,0,0,0\n"  # too narrow
    ),
    "oidv6-validation-annotations-vrd.csv": (
        "ImageID,LabelName1,LabelName2,XMin1,XMax1,YMin1,YMax1,"
        "XMin2,XMax2,YMin2,YMax2,RelationshipLabel\n"
        "d,/m/man,/m/horse,0.2,0.4,0.1,0.5,0.1,0.7,0.3,0.9,ride\n"
        "d,/m/man,/m/run,0.2,0.5,0.1,0.6,0.2,0.5,0.1,0.6,is\n"
    ),
}


@final
class OfflineClient:
    def save(self, url: str, path: Path) -> None:
        raise AssertionError(f"tried to download {url}")


@final
class OpenImagesSourceTest(unittest.TestCase):
    def source(self, cache: Path, **queries: object) -> OpenImagesSource:
        for name, text in FILES.items():
            _ = (cache / name).write_text(text)
        options: dict[str, object] = {
            "classes": ["Dog", "Cattle"],
            "relationships": [("Man", "ride", "Horse"), ("Man", "is", "Run")],
        }
        options.update(queries)
        return OpenImagesSource(
            OfflineClient(),
            cache,
            ["validation"],
            captions={"Cattle": "cow"},
            limit=10,
            min_box=0.2,
            **options,  # pyright: ignore[reportArgumentType]
        )

    def test_indexes_upright_single_object_crops_with_captions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory))
        self.assertEqual(
            source.queries, ["dog", "cow", "man riding horse", "running man"]
        )
        (dog,) = source.records["dog"]
        self.assertEqual(dog.crop, (0.1, 0.2, 0.6, 0.9))
        self.assertEqual(dog.license, "CC BY 2.0")
        self.assertEqual(
            dog.download_url,
            "https://open-images-dataset.s3.amazonaws.com/validation/a.jpg",
        )
        self.assertEqual(source.records["cow"], [])
        (ride,) = source.records["man riding horse"]
        self.assertEqual(ride.crop, (0.1, 0.1, 0.7, 0.9))  # union of both boxes
        self.assertEqual(len(source.records["running man"]), 1)

    def test_rejects_unknown_class_with_suggestion(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "did you mean Dog"),
        ):
            _ = self.source(Path(directory), classes=["Dgo"])

    def test_caption_drops_qualifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory), classes=["Orange (fruit)"])
        self.assertEqual(source.caption("Orange (fruit)"), "orange")
        self.assertEqual(license_name("https://example.com/other"), "")
