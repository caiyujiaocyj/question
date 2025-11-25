
def export_cluster_npz_to_ifc(cluster_npz_path, ifc_out_path, cluster_id=0):
    segs = fit_from_saved_cluster(cluster_npz_path)
    if len(segs) == 0:
        print("No cylinder found!")
        return
    export_single_seg_to_ifc(segs[0], ifc_out_path, seg_id=cluster_id)