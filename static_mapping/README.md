static_mapping
====

1. generate RGB-D data
2. visualize RGB-D data
   ```shell
   python3 rgbd_data_explorer.py --data-file /home/daizhirui/Data/mobile_language_mapping/prepare_groceries/pick/all_static.pt \
        --sensor-names fetch_head fetch_hand --min-depth 0 --max-depth 50 --trajectory-downsample 10 --image-downsample 10
   ```
3. generate clip data
4. train the static mapping model
5. visualize the grid net
    ```shell
    PYTHONPATH=$(pwd)/.. python visualize_grid_net.py \
        visualize_grid_net.yaml \
        cfg.test_model_dir=/home/daizhirui/results/mobile_language_mapping/static_mapping_per_episode/nearest-exact/20250730-095458/best \
        frame_downsample_factor=10 \
        pcd_downsample_factor=10
    ```
