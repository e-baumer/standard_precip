import numpy as np
import pandas as pd
from standard_precip import spi


def test_gamma_lmoments():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="lmom", dist_type="gam"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.678092, 4)

def test_gamma_mle():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="mle", dist_type="gam", floc=0
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.696543, 4)

def test_exp_lmoments():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="lmom", dist_type="exp"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.575136, 4)

def test_exp_mle():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="mle", dist_type="exp"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.177047, 4)

def test_gev_lmoments():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="lmom", dist_type="gev"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.695562, 4)

def test_gev_mle():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="mle", dist_type="gev"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.701048, 4)

def test_gpa_lmoments():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="lmom", dist_type="gpa"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.560434, 4)

def test_gpa_mle():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="mle", dist_type="gpa"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.288574, 4)

def test_gum_lmoments():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="lmom", dist_type="gum"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.698605, 4)

def test_gum_mle():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="mle", dist_type="gum"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.720206, 4)

def test_nor_lmoments():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="lmom", dist_type="nor"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.753621, 4)

def test_nor_mle():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="mle", dist_type="nor"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.712801, 4)

def test_pe3_lmoments():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="lmom", dist_type="pe3"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.652750, 4)

def test_pe3_mle():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="mle", dist_type="pe3"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.670490, 4)

def test_wei_lmoments():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="lmom", dist_type="wei"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.624138, 4)

def test_wei_mle():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="mle", dist_type="wei"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.625167, 4)

def test_glo_mle():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="mle", dist_type="glo"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.721554, 4)

def test_gno_mle():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="mle", dist_type="gno"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.655203, 4)

def test_kap_mle():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="mle", dist_type="kap"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[0], 4) == np.round(-0.312152, 4)

def test_wak_lmoments():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/monthly_data.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'TotalPrecipitation', freq="M", fit_type="lmom", dist_type="wak"
    )
    assert np.round(df_spi['TotalPrecipitation_calculated_index'].iloc[3], 4) == np.round(-0.204953, 4)

def test_3month_spi():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/wichita_rain.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'precip', freq="M", fit_type="lmom", scale=3,
        dist_type="gam"
    )
    assert np.round(df_spi['precip_scale_3_calculated_index'].iloc[2], 4) == np.round(0.856479, 4)

def test_daily_nan():
    new_spi = spi.SPI()
    df_rainfall = pd.read_csv('data/daily_data_test.csv')
    df_spi = new_spi.calculate(
        df_rainfall, 'date', 'precip', freq="D", fit_type="lmom", scale=1,
        dist_type="gam"
    )
    assert np.isnan(df_spi['precip_calculated_index'].iloc[0])
