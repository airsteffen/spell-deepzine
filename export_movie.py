from subprocess import call


def export_movie(output_file, 
                input_file_pattern='interpolation_%05d.png', 
                image_format='image2', 
                framerate=60, 
                output_size=(64, 64), 
                codec='libx264', 
                crf=0, 
                pix_format='yuv420p', 
                ffmpeg_exe='ffmpeg'):
    
    """ A wrapper script for converting images into a video using ffmpeg.
        I couldn't explain the parameters to you if I tried.
        I followed this guide: https://opensource.com/article/17/6/ffmpeg-convert-media-file-formats.
    
    Parameters
    ----------
    output_file : TYPE
        Description
    input_file_pattern : str, optional
        Description
    image_format : str, optional
        Description
    framerate : int, optional
        Description
    output_size : tuple, optional
        Description
    codec : str, optional
        Description
    crf : int, optional
        Description
    pix_format : str, optional
        Description
    ffmpeg_exe : str, optional
        Description
    """
    
    command = [ffmpeg_exe, '-r', str(framerate), '-f', image_format, '-s', 'x'.join(output_size), '-i', '-vcodec', codec, '-crf', str(crf), '-pix_fmt', pix_format, output_file]
    call(' '.join(command), shell=True)


if __name__ == '__main__':

    pass