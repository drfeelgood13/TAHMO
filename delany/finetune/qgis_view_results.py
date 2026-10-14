"""QGIS startup script: test images + labels + one results folder, outlines only.

Usage: qgis --code finetune/qgis_view_results.py      (RESULTS env var picks the folder, default zero_shot)
       ROOT env var points at the local finetune/ folder
"""
import csv
import glob
import os

from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer
from qgis.utils import iface

ROOT = os.environ.get("ROOT", "/home/giani/JIP/cleanDelAny/Delineate-Anything/finetune")
RESULTS = os.environ.get("RESULTS", "zero_shot")

project = QgsProject.instance()
root = project.layerTreeRoot()
with open(os.path.join(ROOT, "yolo_dataset", "splits.csv")) as f:
    test = sorted(row["folder"] for row in csv.DictReader(f) if row["split"] == "test")


def outline(layer, color, width):
    sym = layer.renderer().symbol()
    sym.symbolLayer(0).setBrushStyle(0)  # Qt.NoBrush
    sym.symbolLayer(0).setStrokeColor(__import__("qgis.PyQt.QtGui", fromlist=["QColor"]).QColor(color))
    sym.symbolLayer(0).setStrokeWidth(width)
    layer.triggerRepaint()


labels = QgsVectorLayer(os.path.join(ROOT, "labels", "filtered_grids(1).gpkg"), "labels (fields)", "ogr")
labels.setSubsetString("\"crop\" != 'noncrop'")
project.addMapLayer(labels, False)
outline(labels, "#ffff00", 0.6)

preds = root.insertGroup(1, f"predictions: {RESULTS}")
images = root.insertGroup(2, "test images")
root.insertLayer(0, labels)

for folder in test:
    gpkg = os.path.join(ROOT, "results", RESULTS, folder + ".gpkg")
    if os.path.exists(gpkg):
        layer = QgsVectorLayer(gpkg, folder, "ogr")
        project.addMapLayer(layer, False)
        preds.addLayer(layer)
        outline(layer, "#ff00ff", 0.5)
    tif = glob.glob(os.path.join(ROOT, "DATASET*", "DATASET*", folder, "mosaic_utm.tif"))[0]
    raster = QgsRasterLayer(tif, folder)
    project.addMapLayer(raster, False)
    images.addLayer(raster)

images.setExpanded(False)
preds.setExpanded(False)
first = images.children()[0].layer()
iface.mapCanvas().setDestinationCrs(first.crs())
iface.mapCanvas().setExtent(first.extent())
iface.mapCanvas().refresh()
