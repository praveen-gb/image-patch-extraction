import numpy as np
from skimage import io
import os
np.seterr(divide='ignore', invalid='ignore')  # ignore divide by zero warnings


def patch_extraction_stitching(grayscale_image, config, options):
    '''
    Read input image, extract patches and stitch the patches to reconstruct input image
    '''

    stride_x = round(options.frame_x / 2)  # half padding in x axis (50% overlap)
    stride_y = round(options.frame_y / 2)  # half padding in y axis (50% overlap)
    ny, nx = grayscale_image.shape

    im_sum = np.zeros((ny, nx))
    im_max = np.zeros((ny, nx))
    im_count = np.zeros((ny, nx))
    end_x = 0
    start_x = 0
    tile_num = 0
    while end_x < nx:
        start_y = 0
        end_y = 0
        while end_y < ny:
            end_x = min(start_x + options.frame_x, nx)
            end_y = min(start_y + options.frame_y, ny)
            im_tile = grayscale_image[start_y:end_y, start_x:end_x]
            ty, tx = im_tile.shape

            if ty < options.frame_y:
                im_tile = np.append(im_tile, np.zeros((options.frame_y - ty, tx)), 0)
                ty, tx = im_tile.shape

            if tx < options.frame_x:
                im_tile = np.append(im_tile, np.zeros((options.frame_y, options.frame_x - tx)), 1)
                ty, tx = im_tile.shape

            tx += 1
            ty += 1
            io.imsave(os.path.join(config['output_dir'] + str(tile_num) + '_patch.png'), im_tile.astype('uint8'))

            # stitches the patches to obtain full input image
            end_x = start_x + options.frame_x
            end_y = start_y + options.frame_y

            if end_x > nx:
                tx = nx - start_x
                end_x = nx
            else:
                tx = options.frame_x
            if end_y > ny:
                ty = ny - start_y
                end_y = ny
            else:
                ty = options.frame_y

            im_max[start_y:end_y, start_x:end_x] = np.maximum(
                im_tile[:ty, :tx], im_max[start_y:end_y, start_x:end_x])
            im_sum[start_y:end_y, start_x:end_x] = (im_tile[:ty, :tx] +
                                                    im_sum[start_y:end_y, start_x:end_x])
            im_count[start_y:end_y, start_x:end_x] = (np.ones((ty, tx)) +
                                                      im_count[start_y:end_y, start_x:end_x])

            tile_num += 1
            start_y += stride_y
        start_x += stride_x
        im_mean = im_sum / im_count  # mean probability
        reconstructed_image_path = os.path.join(config['output_dir'] + 'reconstructed_image/')
        os.makedirs(reconstructed_image_path, exist_ok=True)
        io.imsave(os.path.join(reconstructed_image_path + "mean_probability_reconstructed.png"), im_mean.astype('uint8'))
        io.imsave(os.path.join(reconstructed_image_path + "max_probability_reconstructed.png"),
                  im_max.astype('uint8'))
