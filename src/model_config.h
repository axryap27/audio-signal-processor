#pragma once

/*
 * model_config.h — normalization parameters and genre labels
 *
 * These values come from model_metadata.json, which was produced by
 * train_model.py's StandardScaler.  Every raw feature must be normalized
 * as:  (value - mean) / scale  before being fed to the TFLite model.
 */

#include <Arduino.h>

static const int NUM_FEATURES = 48;
static const int NUM_GENRES   = 10;

static const char* GENRE_LABELS[NUM_GENRES] = {
    "blues", "classical", "country", "disco", "hip-hop",
    "jazz", "metal", "pop", "reggae", "rock"
};

/*
 * Feature order (must match the training pipeline):
 *   [0]  centroid          [16] band_9           [32] mfcc_6
 *   [1]  spread            [17] band_10          [33] mfcc_7
 *   [2]  flatness          [18] band_11          [34] mfcc_8
 *   [3]  rolloff           [19] band_12          [35] mfcc_9
 *   [4]  zcr               [20] band_13          [36] mfcc_10
 *   [5]  rms_energy        [21] band_14          [37] mfcc_11
 *   [6]  peak_amplitude    [22] band_15          [38] mfcc_12
 *   [7]  band_0            [23] mfcc_0           [39] chroma_0
 *   [8]  band_1            [24] mfcc_1           [40] chroma_1
 *   [9]  band_2            [25] mfcc_2           [41] chroma_2
 *   [10] band_3            [26] mfcc_3           [42] chroma_3
 *   [11] band_4            [27] mfcc_4           [43] chroma_4
 *   [12] band_5            [28] mfcc_5           [44] chroma_5
 *   [13] band_6            [29] mfcc_6           [45] chroma_6 ... etc
 *   [14] band_7
 *   [15] band_8
 */

static const float SCALER_MEAN[NUM_FEATURES] = {
    2064.953098864953,       // centroid
    2302.0051427739986,      // spread
    0.004073880613675804,    // flatness
    4282.081715041894,       // rolloff
    0.054610856228095646,    // zcr
    0.1286352030980981,      // rms_energy
    0.8421317497467468,      // peak_amplitude
    26.736331955205205,      // band_0
    23.991527278218218,      // band_1
    17.38655729183183,       // band_2
    13.60671074938939,       // band_3
    9.933632058328326,       // band_4
    7.492204776966967,       // band_5
    5.799203118518519,       // band_6
    4.302931065415416,       // band_7
    3.24647774586987,        // band_8
    2.6482775627287287,      // band_9
    2.2301063499144145,      // band_10
    1.6931575579649651,      // band_11
    1.1056738874343344,      // band_12
    0.7684486515129529,      // band_13
    0.32926803388094095,     // band_14
    3.203528512508509e-05,   // band_15
    -216.42149822722723,     // mfcc_0
    172.6644585485486,       // mfcc_1
    -47.56180108677978,      // mfcc_2
    51.645700616997,         // mfcc_3
    0.8497569926330334,      // mfcc_4
    7.795996790044044,       // mfcc_5
    13.220107837415416,      // mfcc_6
    -9.755291732887587,      // mfcc_7
    14.76532305187888,       // mfcc_8
    -6.812702899893713,      // mfcc_9
    2.6433087275175176,      // mfcc_10
    4.088740136939941,       // mfcc_11
    -5.315293459219219,      // mfcc_12
    0.46227069851351354,     // chroma_0
    0.440833525009009,       // chroma_1
    0.4596124231381381,      // chroma_2
    0.45230049496996994,     // chroma_3
    0.46402399303703706,     // chroma_4
    0.4550546686726727,      // chroma_5
    0.4330101217317317,      // chroma_6
    0.4546196718318318,      // chroma_7
    0.4509176556496497,      // chroma_8
    0.46901961641641643,     // chroma_9
    0.4529461868518519,      // chroma_10
    0.45115200714714715      // chroma_11
};

static const float SCALER_SCALE[NUM_FEATURES] = {
    664.634028802691,        // centroid
    532.4797419079133,       // spread
    0.0021798668120544466,   // flatness
    1474.004653419889,       // rolloff
    0.02260303666402458,     // zcr
    0.06445793521370333,     // rms_energy
    0.32246549920843953,     // peak_amplitude
    21.750194228736795,      // band_0
    16.802034939215183,      // band_1
    11.380662259401792,      // band_2
    7.804762551171848,       // band_3
    5.425677013573079,       // band_4
    4.030258443980563,       // band_5
    3.2057001394359403,      // band_6
    2.6712539900374774,      // band_7
    2.0388129784234317,      // band_8
    1.7813888874110173,      // band_9
    1.6718911623773702,      // band_10
    1.3391212714351697,      // band_11
    0.9860016289603656,      // band_12
    0.7085897879681093,      // band_13
    0.47226983879485607,     // band_14
    3.7017276175242473e-05,  // band_15
    87.47086624988253,       // mfcc_0
    21.95470400209063,       // mfcc_1
    30.60137809690126,       // mfcc_2
    20.993733643558034,      // mfcc_3
    14.888155014311273,      // mfcc_4
    10.604093663229762,      // mfcc_5
    9.535123277619867,       // mfcc_6
    7.134007930157333,       // mfcc_7
    9.132990650078444,       // mfcc_8
    7.3403775370087585,      // mfcc_9
    8.702756470776638,       // mfcc_10
    6.747235639385816,       // mfcc_11
    5.0885113343547586,      // mfcc_12
    0.1115779028191679,      // chroma_0
    0.11610344269972227,     // chroma_1
    0.1154508633490534,      // chroma_2
    0.11463299261225211,     // chroma_3
    0.11407946830962236,     // chroma_4
    0.11150874743969368,     // chroma_5
    0.10987349626956829,     // chroma_6
    0.10537662540739418,     // chroma_7
    0.10815772600005759,     // chroma_8
    0.10839337615762021,     // chroma_9
    0.10726359605223264,     // chroma_10
    0.11036864993596851      // chroma_11
};
