"""Класс DataProcessor: загрузка, предобработка и анализ данных Яндекс Афиши."""
import os

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine
from statsmodels.stats.proportion import proportions_ztest
from phik import phik_matrix

ORDERS_QUERY = '''
SELECT p.user_id,
       p.device_type_canonical,
       p.order_id,
       p.created_dt_msk  AS order_dt,
       p.created_ts_msk  AS order_ts,
       p.currency_code,
       p.revenue,
       p.tickets_count,
       p.created_dt_msk::date
           - LAG(p.created_dt_msk::date) OVER (
               PARTITION BY p.user_id
               ORDER BY p.created_dt_msk
           )             AS days_since_prev,
       p.event_id,
       e.event_type_main,
       p.service_name,
       r.region_name,
       c.city_name
FROM afisha.purchases AS p
INNER JOIN afisha.events AS e ON p.event_id = e.event_id
INNER JOIN afisha.city AS c ON e.city_id = c.city_id
INNER JOIN afisha.regions AS r ON c.region_id = r.region_id
WHERE p.device_type_canonical IN ('mobile', 'desktop')
  AND e.event_type_main != 'фильм'
ORDER BY p.user_id
'''

DUP_COLS = ['user_id', 'order_ts', 'device_type_canonical', 'service_name', 'event_id', 'revenue']
CATEGORY_COLS = ['device_type_canonical', 'currency_code', 'event_type_main',
                  'service_name', 'region_name', 'city_name']

WEEKDAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
WEEKDAY_NAMES_RU = {
    'Monday': 'Понедельник', 'Tuesday': 'Вторник', 'Wednesday': 'Среда',
    'Thursday': 'Четверг', 'Friday': 'Пятница', 'Saturday': 'Суббота', 'Sunday': 'Воскресенье',
}

def _order_segment(n):
    if n == 1:
        return '1 заказ'
    if n == 2:
        return '2 заказа'
    if 3 <= n <= 4:
        return '3-4 заказа'
    return '5+ заказов'

class DataProcessor:
    """Загружает заказы Яндекс Афиши, приводит их к единому виду и считает
    метрики лояльности: профили пользователей, retention по сегментам и
    корреляцию признаков с числом заказов.
    """

    def __init__(self, raw_data_path='data/orders_raw.csv',
                 tenge_path='data/final_tickets_tenge_df.csv',
                 revenue_percentile=0.99, orders_percentile=0.99):
        self.raw_data_path = raw_data_path
        self.tenge_path = tenge_path
        self.revenue_percentile = revenue_percentile
        self.orders_percentile = orders_percentile

        self.raw_data = None
        self.data = None
        self.profile = None

    def extract_from_db(self):
        """Выгружает заказы из БД data-analyst-afisha и сохраняет CSV в data/.

        Доступы к БД читаются из .env. Метод нужен один раз, чтобы получить
        сырой файл для load_orders_data(); сам пайплайн его не вызывает.
        """
        connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(
            os.getenv('DB_USER'), os.getenv('DB_PWD'),
            os.getenv('DB_HOST'), os.getenv('DB_PORT'),
            os.getenv('DB_NAME')
        )
        engine = create_engine(connection_string)
        raw_data = pd.read_sql_query(ORDERS_QUERY, con=engine)
        raw_data.to_csv(self.raw_data_path, index=False)
        return raw_data

    def load_orders_data(self):
        """Читает сырые данные о заказах из CSV, ранее выгруженного через SQL."""
        self.raw_data = pd.read_csv(
            self.raw_data_path,
            parse_dates=['order_dt', 'order_ts']
        )
        return self.raw_data

    def load_currency_data(self):
        """Читает курс казахстанского тенге к рублю из локального CSV-файла."""
        tenge = pd.read_csv(self.tenge_path, parse_dates=['data'])
        return tenge.rename(columns={'data': 'order_dt'})

    def preprocess(self):
        """Приводит валюту к рублям, чистит дубликаты, типы и выбросы."""
        data = self.raw_data.drop_duplicates(subset=DUP_COLS, keep='first').reset_index(drop=True)

        tenge = self.load_currency_data()
        data = data.merge(tenge[['order_dt', 'curs', 'nominal']], on='order_dt', how='left')
        data['revenue_rub'] = np.where(
            data['currency_code'] == 'kzt',
            data['revenue'] * data['curs'] / data['nominal'],
            data['revenue'],
        )
        data = data.drop(columns=['curs', 'nominal'])

        data['order_dt'] = pd.to_datetime(data['order_dt'])
        data['order_ts'] = pd.to_datetime(data['order_ts'])
        data[CATEGORY_COLS] = data[CATEGORY_COLS].astype('category')
        data['order_id'] = data['order_id'].astype('int32')
        data['event_id'] = data['event_id'].astype('int32')
        data['tickets_count'] = data['tickets_count'].astype('int16')
        data['revenue'] = data['revenue'].astype('float32')
        data['revenue_rub'] = data['revenue_rub'].astype('float32')

        data = data[data['revenue_rub'] >= 0].reset_index(drop=True)
        revenue_cap = data['revenue_rub'].quantile(self.revenue_percentile)
        data = data[data['revenue_rub'] <= revenue_cap].reset_index(drop=True)

        data = data.sort_values(['user_id', 'order_ts']).reset_index(drop=True)
        data['days_since_prev'] = (data.groupby('user_id')['order_dt']
                                    .diff().dt.days.astype('float32'))

        self.data = data
        return self.data

    def build_user_profiles(self):
        """Строит профиль пользователя и отсекает выбросы по числу заказов."""
        profile = self.data.groupby('user_id').agg(
            first_order_dt=('order_dt', 'first'),
            last_order_dt=('order_dt', 'last'),
            first_device=('device_type_canonical', 'first'),
            first_region=('region_name', 'first'),
            first_event_type=('event_type_main', 'first'),
            total_orders=('order_id', 'count'),
            avg_revenue_rub=('revenue_rub', 'mean'),
            avg_days_between=('days_since_prev', 'mean'),
        ).reset_index()

        profile['is_two'] = profile['total_orders'] >= 2

        orders_cap = profile['total_orders'].quantile(self.orders_percentile)
        profile = profile[profile['total_orders'] <= orders_cap].reset_index(drop=True)

        profile['first_order_weekday'] = profile['first_order_dt'].dt.day_name()
        profile['orders_segment'] = profile['total_orders'].apply(_order_segment)

        self.profile = profile
        return self.profile

    def calculate_retention(self):
        """Считает retention по сегментам и проверяет продуктовые гипотезы."""
        profile = self.profile
        overall_return_rate = profile['is_two'].mean()

        segments = {}
        for name, group_col in [('event_type', 'first_event_type'),
                                 ('device', 'first_device'),
                                 ('region', 'first_region')]:
            seg = profile.groupby(group_col, observed=True).agg(
                n_users=('user_id', 'count'),
            ).sort_values('n_users', ascending=False)
            seg['share'] = seg['n_users'] / seg['n_users'].sum()
            seg['return_rate'] = profile.groupby(group_col, observed=True)['is_two'].mean()
            segments[name] = seg

        sport_concert = profile[profile['first_event_type'].isin(['спорт', 'концерты'])]
        crosstab_h1 = pd.crosstab(sport_concert['first_event_type'], sport_concert['is_two'])
        chi2_event, p_value_event, _, _ = stats.chi2_contingency(crosstab_h1)
        is_concert = profile['first_event_type'] == 'концерты'
        is_sport = profile['first_event_type'] == 'спорт'
        hypothesis_event_type = {
            'chi2': chi2_event,
            'p_value': p_value_event,
            'return_rate_concert': profile.loc[is_concert, 'is_two'].mean(),
            'return_rate_sport': profile.loc[is_sport, 'is_two'].mean(),
        }

        seg_region = segments['region']
        seg_region_stable = seg_region[seg_region['n_users'] >= 30]
        stable_n_users = seg_region_stable['n_users']
        stable_return_rate = seg_region_stable['return_rate']
        corr_spearman = stable_n_users.corr(stable_return_rate, method='spearman')
        corr_pearson = stable_n_users.corr(stable_return_rate, method='pearson')

        top10_ids = seg_region.sort_values('n_users', ascending=False).head(10).index
        top_mask = profile['first_region'].isin(top10_ids)
        successes = [profile.loc[top_mask, 'is_two'].sum(), profile.loc[~top_mask, 'is_two'].sum()]
        nobs = [top_mask.sum(), (~top_mask).sum()]
        z_stat, p_value_regions = proportions_ztest(successes, nobs)
        hypothesis_region_size = {
            'z_stat': z_stat,
            'p_value': p_value_regions,
            'return_rate_top10': profile.loc[top_mask, 'is_two'].mean(),
            'return_rate_rest': profile.loc[~top_mask, 'is_two'].mean(),
            'corr_spearman': corr_spearman,
            'corr_pearson': corr_pearson,
        }

        weekday_stats = profile.groupby('first_order_weekday').agg(
            n_users=('user_id', 'count'),
            return_rate=('is_two', 'mean'),
        ).reindex(WEEKDAY_ORDER)
        weekday_stats.index = weekday_stats.index.map(WEEKDAY_NAMES_RU)

        crosstab_weekday = pd.crosstab(profile['first_order_weekday'], profile['is_two'])
        chi2_weekday, p_value_weekday, _, _ = stats.chi2_contingency(crosstab_weekday)

        return {
            'overall_return_rate': overall_return_rate,
            'segments': segments,
            'hypothesis_event_type': hypothesis_event_type,
            'hypothesis_region_size': hypothesis_region_size,
            'weekday': {
                'stats': weekday_stats,
                'chi2': chi2_weekday,
                'p_value': p_value_weekday,
            },
        }

    def correlation_analysis(self):
        """Считает phi_k корреляцию признаков первого заказа с числом заказов."""
        profile = self.profile

        phik_cols = ['first_device', 'first_region', 'first_event_type',
                     'first_order_weekday', 'avg_revenue_rub', 'total_orders']
        interval_cols = ['avg_revenue_rub', 'total_orders']
        phik_matrix_res = phik_matrix(profile[phik_cols], interval_cols=interval_cols)

        seg_cols = ['first_device', 'first_region', 'first_event_type',
                    'first_order_weekday', 'avg_revenue_rub', 'orders_segment']
        phik_matrix_seg = phik_matrix(profile[seg_cols], interval_cols=['avg_revenue_rub'])

        returning_profile = profile[profile['total_orders'] >= 2].copy()
        history_cols = ['avg_days_between', 'total_orders']
        phik_history = returning_profile[history_cols].phik_matrix(interval_cols=history_cols)

        return {
            'phik_matrix': phik_matrix_res,
            'phik_matrix_segmented': phik_matrix_seg,
            'phik_history': phik_history,
        }

    def run(self):
        """Запускает полный пайплайн: загрузка -> предобработка -> анализ."""
        self.load_orders_data()
        self.preprocess()
        self.build_user_profiles()
        retention = self.calculate_retention()
        correlation = self.correlation_analysis()
        return {
            'data': self.data,
            'profile': self.profile,
            'retention': retention,
            'correlation': correlation,
        }