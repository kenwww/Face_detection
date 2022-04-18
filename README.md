# Face_detection
def Face_detection(Img, image_size):
    minsize = 20
    threshold = [ 0.6, 0.7, 0.7 ]  
    factor = 0.709 
    margin = 44
    gpu_memory_fraction = 1.0

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

            Img_size = np.asarray(Img.shape)[0:2]
            bounding_boxes, _ = align.detect_face.detect_face(Img, minsize, pnet, rnet, onet, threshold, factor)
            faces = np.zeros((len(bounding_boxes), image_size, image_size, 3), dtype = "uint8")
            bb = np.zeros((len(bounding_boxes), 4), dtype=np.int32)
            for i in range(len(bounding_boxes)):            
                det = np.squeeze(bounding_boxes[i,0:4])
                bb[i, 0] = np.maximum(det[0]-margin/2, 0)
                bb[i, 1] = np.maximum(det[1]-margin/2, 0)
                bb[i, 2] = np.minimum(det[2]+margin/2, Img_size[1])
                bb[i, 3] = np.minimum(det[3]+margin/2, Img_size[0])
                cropped = Img[bb[i, 1]:bb[i, 3],bb[i, 0]:bb[i, 2],:]
                aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                faces[i, :, :, :] = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return faces, bb
