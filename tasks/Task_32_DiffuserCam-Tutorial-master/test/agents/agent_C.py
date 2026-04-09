def C(M_arr, full_size, sensor_size):
    # Crop operator
    top = (full_size[0] - sensor_size[0]) // 2
    left = (full_size[1] - sensor_size[1]) // 2
    return M_arr[top:top+sensor_size[0], left:left+sensor_size[1]]
