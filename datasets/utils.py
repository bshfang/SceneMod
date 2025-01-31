import numpy as np

def scale(x, minimum, maximum):
    X = x.astype(np.float32)
    X = np.clip(X, minimum, maximum)
    X = ((X - minimum) / (maximum - minimum))
    X = 2 * X - 1
    return X


def descale(x, minimum, maximum):
    x = (x + 1) / 2
    x = x * (maximum - minimum) + minimum
    return x



def get_resize_scale_bound(translations, bound):
    """
    translations: n,3
    bound: min(3), max(3)
    """
    scale_min_values = []
    dims = [0,2]
    for dim in dims:
        min_bound, max_bound = bound[0][dim], bound[1][dim]
        points_dim = translations[:, dim]

        for point in points_dim:
            if point < 0:
                scale_min_values.append(min_bound / point)
            elif point > 0:
                scale_min_values.append(max_bound / point)
    final_scale_min = min(scale_min_values)
    
    return final_scale_min



def get_trans_range_bound(translations, bound):
    """
    translations: n,3
    bound: min(3), max(3)
    """

    translation_range = np.zeros((3, 2)) 

    for dim in range(3): 
        points_min = np.min(translations[:, dim])
        points_max = np.max(translations[:, dim])

        translation_min = np.min(bound[0][dim] - points_min, 0)
        translation_max = np.max(bound[1][dim] - points_max, 0)

        translation_range[dim, :] = [translation_min, translation_max]

    return translation_range[0], translation_range[2]  # y,x in the data