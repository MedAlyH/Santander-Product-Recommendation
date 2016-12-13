import pandas as pd
import os


def read_data(filename):
    df = pd.read_csv(filename, dtype={"sexo": str, "ind_nuevo": str,
                                      "indext": str},
                     skipinitialspace=True)
    return df

def clean_indrel_lmes(arr):
    arr = arr.fillna(-1)
    for i, item in enumerate(arr):
        if item == 'P':
            arr[i] = 5
    arr = pd.to_numeric(arr)
    arr = arr.astype(int)
    return arr

def clean_data(filename):
    df = read_data(filename)
    df['indrel_1mes'] = clean_indrel_lmes(df['indrel_1mes'])

    mapping_ind_empleado = {'N': 1, 'B': 2, 'F': 3, 'A': 4, 'S': 5}
    df['ind_empleado'] = df['ind_empleado'].apply(lambda x:
                                                  mapping_ind_empleado[x]
                                                  if x in mapping_ind_empleado
                                                  else -1)

    mapping_pais = {'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117,
                    'BO': 62, 'JP': 82, 'JM': 116, 'BR': 17, 'BY': 64,
                    'BZ': 113, 'RU': 43, 'RS': 89, 'RO': 41, 'GW': 99,
                    'GT': 44, 'GR': 39, 'GQ': 73, 'HN': 22, 'MT': 118,
                    'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96,
                    'GH': 88, 'OM': 100, 'HR': 67, 'HU': 106, 'HK': 34,
                    'AD': 35, 'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20,
                    'PK': 84, 'PH': 91, 'PL': 30, 'EE': 52, 'EG': 74, 'ZA': 75,
                    'EC': 19, 'AL': 25, 'VN': 90, 'ET': 54, 'ZW': 114, 'ES': 1,
                    'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15,
                    'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38,
                    'FI': 23, 'NI': 33, 'NL': 7, 'NO': 46, 'NG': 83, 'NZ': 93,
                    'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4,
                    'CA': 2, 'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36,
                    'CU': 72, 'KE': 65, 'KH': 95, 'SV': 53, 'SK': 69, 'KR': 87,
                    'KW': 92, 'SN': 47, 'SL': 97, 'KZ': 111, 'SA': 56,
                    'SE': 24, 'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10,
                    'LT': 103, 'LU': 59, 'TH': 79, 'TG': 86, 'LY': 108,
                    'AE': 37,  'CR': 32, 'SG': 66, 'DZ': 80, 'AU': 63,
                    'VE': 14, 'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13,
                    'AT': 6, 'IN': 31, 'IE': 5, 'QA': 58, 'MZ': 27, 'MK': 105,
                    'LB': 81, 'TW': 29, 'TR': 70, 'TN': 85
                    }
    df['pais_residencia'] = df['pais_residencia'].apply(lambda x:
                                                        mapping_pais[x]
                                                        if x in mapping_pais
                                                        else -1)

    mapping_dict = {'H': 1, 'V': 2}
    df['sexo'] = df['sexo'].apply(lambda x: mapping_dict.get(x, -1))

    df['age'] = df['age'].fillna(-1)
    df['age'] = df['age'].astype(int)

    df['antiguedad'] = df['antiguedad'].fillna(-1)

    df['ind_nuevo'] = df['ind_nuevo'].fillna(-1)
    df['ind_nuevo'] = df['ind_nuevo'].astype(int)

    df['indrel'] = df['indrel'].fillna(-1)
    df['indrel'] = df['indrel'].astype(int)

    mapping_triprel = {'I': 1, 'A': 2, 'P': 3, 'R': 4, 'N': 5}
    df['tiprel_1mes'] = df['tiprel_1mes'].apply(lambda x:
                                                mapping_triprel.get(x, -1))

    mapping_SN = {'S': 1, 'N':2}
    df['indresi'] = df['indresi'].apply(lambda x: mapping_SN.get(x, -1))

    df['indext'] = df['indext'].apply(lambda x: mapping_SN.get(x, -1))

    df['conyuemp'] = df['conyuemp'].apply(lambda x: mapping_SN.get(x, -1))
    mapping_canal_entrada = {'013': 49, 'KHP': 160, 'KHQ': 157, 'KHR': 161,
                             'KHS': 162, 'KHK': 10, 'KHL': 6, 'KHM': 12,
                             'KHN': 21, 'KHO': 13, 'KHA': 22, 'KHC': 9,
                             'KHD': 2, 'KHE': 1, 'KHF': 19, '025': 159,
                             'KAC': 57, 'KAB': 28, 'KAA': 39, 'KAG': 26,
                             'KAF': 23, 'KAE': 30, 'KAD': 16, 'KAK': 51,
                             'KAJ': 41, 'KAI': 35, 'KAH': 31, 'KAO': 94,
                             'KAN': 110, 'KAM': 107, 'KAL': 74, 'KAS': 70,
                             'KAR': 32, 'KAQ': 37, 'KAP': 46, 'KAW': 76,
                             'KAV': 139, 'KAU': 142, 'KAT': 5, 'KAZ': 7,
                             'KAY': 54, 'KBJ': 133, 'KBH': 90, 'KBN': 122,
                             'KBO': 64, 'KBL': 88, 'KBM': 135, 'KBB': 131,
                             'KBF': 102, 'KBG': 17, 'KBD': 109, 'KBE': 119,
                             'KBZ': 67, 'KBX': 116, 'KBY': 111, 'KBR': 101,
                             'KBS': 118, 'KBP': 121, 'KBQ': 62, 'KBV': 100,
                             'KBW': 114, 'KBU': 55, 'KCE': 86, 'KCD': 85,
                             'KCG': 59, 'KCF': 105, 'KCA': 73, 'KCC': 29,
                             'KCB': 78, 'KCM': 82, 'KCL': 53, 'KCO': 104,
                             'KCN': 81, 'KCI': 65, 'KCH': 84, 'KCK': 52,
                             'KCJ': 156, 'KCU': 115, 'KCT': 112, 'KCV': 106,
                             'KCQ': 154, 'KCP': 129, 'KCS': 77, 'KCR': 153,
                             'KCX': 120, 'RED': 8, 'KDL': 158, 'KDM': 130,
                             'KDN': 151, 'KDO': 60, 'KDH': 14, 'KDI': 150,
                             'KDD': 113, 'KDE': 47, 'KDF': 127, 'KDG': 126,
                             'KDA': 63, 'KDB': 117, 'KDC': 75, 'KDX': 69,
                             'KDY': 61, 'KDZ': 99, 'KDT': 58, 'KDU': 79,
                             'KDV': 91, 'KDW': 132, 'KDP': 103, 'KDQ': 80,
                             'KDR': 56, 'KDS': 124, 'K00': 50, 'KEO': 96,
                             'KEN': 137, 'KEM': 155, 'KEL': 125, 'KEK': 145,
                             'KEJ': 95, 'KEI': 97, 'KEH': 15, 'KEG': 136,
                             'KEF': 128, 'KEE': 152, 'KED': 143, 'KEC': 66,
                             'KEB': 123, 'KEA': 89, 'KEZ': 108, 'KEY': 93,
                             'KEW': 98, 'KEV': 87, 'KEU': 72, 'KES': 68,
                             'KEQ': 138, 'KFV': 48, 'KFT': 92, 'KFU': 36,
                             'KFR': 144, 'KFS': 38, 'KFP': 40, 'KFF': 45,
                             'KFG': 27, 'KFD': 25, 'KFE': 148, 'KFB': 146,
                             'KFC': 4, 'KFA': 3, 'KFN': 42, 'KFL': 34,
                             'KFM': 141, 'KFJ': 33, 'KFK': 20, 'KFH': 140,
                             'KFI': 134, '007': 71, '004': 83, 'KGU': 149,
                             'KGW': 147, 'KGV': 43, 'KGY': 44, 'KGX': 24,
                             'KGC': 18, 'KGN': 11}
    df['canal_entrada'] = df['canal_entrada'].apply(lambda x: mapping_canal_entrada.get(x, -1))

    df['indfall'] = df['indfall'].apply(lambda x: mapping_SN.get(x, -1))

    df['tipodom'] = df['tipodom'].fillna(-1)
    df['tipodom'] = df['tipodom'].astype(int)

    df['cod_prov'] = df['cod_prov'].fillna(-1)
    df['cod_prov'] = df['cod_prov'].astype(int)

    df = df.drop('nomprov', axis=1)

    df['ind_actividad_cliente'] = df['ind_actividad_cliente'].fillna(-1)
    df['ind_actividad_cliente'] = df['ind_actividad_cliente'].astype(int)

    df['renta'] = df['renta'].fillna(-1)

    mapping_segmento = {'01 - TOP': 1,
                        '02 - PARTICULARES': 2,
                        '03 - UNIVERSITARIO': 3
                        }
    df['segmento'] = df['segmento'].apply(lambda x: mapping_segmento.get(x, -1))

    return df


def create_clean_data(datapath):
    train_file = 'train_ver2.csv'
    test_file = 'test_ver2.csv'
    print 'cleaning train set'
    train = clean_data(os.path.join(datapath, train_file))
    train['ind_nomina_ult1'] = train['ind_nomina_ult1'].fillna(0)
    train['ind_nomina_ult1'] = train['ind_nomina_ult1'].astype(int)
    train['ind_nom_pens_ult1'] = train['ind_nom_pens_ult1'].fillna(0)
    train['ind_nom_pens_ult1'] = train['ind_nom_pens_ult1'].astype(int)
    print '-'*30
    print 'cleaning train set'
    test = clean_data(os.path.join(datapath, test_file))
    print '-'*30
    print 'writing'
    train.to_csv(os.path.join(datapath, 'train.csv'), index=False)
    test.to_csv(os.path.join(datapath, 'test.csv'), index=False)


if __name__ == '__main__':
    data_path = '../data/input'
    trainfile = 'train.csv'
    testfile = 'test.csv'
    create_clean_data(data_path)
