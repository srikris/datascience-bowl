import graphlab as gl
import PIL
from PIL import *
from PIL import Image
import StringIO

def multi_class_log_loss(model, sf):
    """
    Compute multi-class log loss given a model and an SFrame.
    """
    prob = model.predict_topk(sf, k=121)
    val = prob.join(
            gl.SFrame({'class': sf['class']}).add_row_number('row_id'), 
            on=['row_id', 'class'])
    val['log-score'] = val['score'].apply(lambda x: math.log(x))
    return -val['log-score'].mean()


def make_submission(model, test):
    """
    Make a submission in the format as asked by Kaggle.
    """
    preds = gl.SFrame({'image': test['path'].apply(lambda x: x.split('/')[-1])})
    preds.add_columns(model.predict_topk(test, k=121)\
        .unstack(['class', o_type], 'dict')\
        .unpack('dict', '')\
        .remove_column('row_id'))
    preds.save('submission.csv')



def from_pil_image(pil_img):
    """
    Convert a PIL image to a Graphlab Image.
    """
    height = pil_img.size[1]
    width = pil_img.size[0]
    if pil_img.mode == 'L':
        image_data = bytearray([z for z in pil_img.getdata()])
        channels = 1
    elif pil_img.mode == 'RGB':
        image_data = bytearray([z for l in pil_img.getdata() for z in l ])
        channels = 3
    else:
         image_data = bytearray([z for l in pil_img.getdata() for z in l])
         channels = 4
    image_data_size = len(image_data)
    return gl.Image(_image_data=image_data, 
                    _width=width, 
                    _height=height, 
                    _channels=channels, 
                    _format_enum=2, 
                    _image_data_size=image_data_size)

def rotate_image(gl_img, angle):
    """
    Rotate an image Pillow.
    """
    img = Image.open(StringIO.StringIO(gl_img._image_data))
    img = img.rotate(angle)
    return from_pil_image(img)


def random_rotate(x):
    """
    Random rotate an image.
    """
    a = x['id'] % 4
    if a == 0:
        return rotate_image(x['image'], 90)
    elif a == 1:
        return rotate_image(x['image'], 180)
    elif a == 2:
        return rotate_image(x['image'], 270)
    elif a == 3:
        return x['image']


if __name__ == "__main__":

    # Assume that you have a directory with the train images
    # Note: The function automatically goes through all the images in your 
    # folder (recursively) and shuffles them and then saves them into an 
    # SFrame.
    print "Loading images..."
    train = gl.image_analysis.load_images('train')

    # Reize the test and train data.
    print "Resizing images..."
    train['image'] = gl.image_analysis.resize(train['image'], 64, 64, 3)
    train['class'] = train['path'].apply(lambda x: x.split('/')[-2])


    # Save (in a compressed format)
    # train.save('train.gl')

    # Perform the data augmentation by making 4 copies of the data.
    print "Data Augmentation..."
    train = train.append(train)
    train = train.append(train)
    train = train.add_row_number()
    train['image-aug'] = train[['id', 'image']].apply(random_rotate)


    # HACK: Create a random split for a validation set to make sure that the 
    # classes are equally balanced in the train and validation set.
    train, valid = gl.recommender.util.random_split_by_user(train, 
                   user_id='class', item_id='image', item_test_proportion=0.1)


    # Load the GL network and train a model
    print "Training Model..."
    network = gl.deeplearning.load('network.conf')
    model = gl.neuralnet_classifier.create(train, 'class', 
                                           features=['image-aug'], 
                                           max_iterations=50, 
                                           network=network, 
                                           validation_set=valid, 
                                           random_mirror=True, 
                                           random_crop=True)

    # Evaluate the model
    print "Score on the validation set: %s" % multi_class_log_loss(model, valid)

    ## Make a submission
    #print "Creating submission..."
    #make_submission(model, test)
    #test = gl.image_analysis.load_images('test')
    #test['image'] = gl.image_analysis.resize(test['image'], 64, 64, 3)




