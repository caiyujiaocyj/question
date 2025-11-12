Traceback (most recent call last):
  File "/media/samsung/samsung/mc.wei/code/fitting_latest/2222.py", line 816, in <module>
    main(parse_args())
  File "/media/samsung/samsung/mc.wei/code/fitting_latest/2222.py", line 811, in main
    vis_single_block(path_pc, args)
  File "/media/samsung/samsung/mc.wei/code/fitting_latest/2222.py", line 803, in vis_single_block
    vis_compare_2_pc(vis_custom, [pc_fitted, lst_pred], [pc_fitted, lst_gt_fn], path_vis, show_ids=True)
  File "/media/samsung/samsung/mc.wei/code/fitting_latest/2222.py", line 749, in vis_compare_2_pc
    func_vis(lst_data1, vis)
  File "/media/samsung/samsung/mc.wei/code/fitting_latest/2222.py", line 590, in vis_custom
    pts = pcd['pts']
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
