"""shapekit - Metashape-style photogrammetry for photos and videos.

Python API (each step mirrors a Metashape Workflow command):

    from shapekit import Project, import_video, align_photos, build_depth_maps, build_point_cloud, \
        build_mesh, build_texture, export_model, generate_report
    p = Project.create("projects/site"); ch = p.chunk()
    import_video(ch, "walk.mp4"); p.save()
    align_photos(ch); build_depth_maps(ch, "medium"); build_point_cloud(ch)
    build_mesh(ch); build_texture(ch); p.save()
    export_model(ch, "site.glb"); generate_report(ch)
"""
__version__ = "0.1.0"

from .project import Project, Chunk, Camera, Sensor  # noqa: F401


def __getattr__(name):
    # lazy imports keep `python -m shapekit --help` fast
    import importlib
    table = {
        "import_video": "video", "add_photos": "video", "estimate_image_quality": "video",
        "align_photos": "sfm", "optimize_cameras": "sfm",
        "build_depth_maps": "mvs", "build_point_cloud": "mvs",
        "build_mesh": "mesh", "build_texture": "texture",
        "export_model": "export", "export_points": "export", "export_cameras": "export",
        "generate_report": "report",
    }
    if name in table:
        return getattr(importlib.import_module("." + table[name], __name__), name)
    raise AttributeError(name)
