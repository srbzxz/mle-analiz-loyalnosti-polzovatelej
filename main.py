"""Точка входа: анализ лояльности пользователей Яндекс Афиши."""
import sys

from dotenv import load_dotenv

from src.processing import DataProcessor

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()


def print_report(results):
    profile = results['profile']
    retention = results['retention']
    correlation = results['correlation']

    n_users = profile.shape[0]
    overall_return_rate = retention['overall_return_rate']
    hyp_event = retention['hypothesis_event_type']
    hyp_region = retention['hypothesis_region_size']
    weekday = retention['weekday']

    print('=' * 70)
    print('ОТЧЁТ: АНАЛИЗ ЛОЯЛЬНОСТИ ПОЛЬЗОВАТЕЛЕЙ ЯНДЕКС АФИШИ')
    print('=' * 70)

    print(f'\nПользователей в выборке: {n_users}')
    print(f'Доля пользователей с 2+ заказами: {overall_return_rate:.2%}')
    print(f"Медиана числа заказов: {profile['total_orders'].median():.0f}")

    print('\n--- Сегменты первого заказа (топ-3 по числу пользователей) ---')
    for name, seg in retention['segments'].items():
        top = seg.sort_values('n_users', ascending=False).head(3)
        print(f'\nПо признаку "{name}":')
        for idx, row in top.iterrows():
            print(f"  {idx}: {row['n_users']} польз. "
                  f"({row['share']:.1%}), return_rate={row['return_rate']:.2%}")

    print('\n--- Проверка гипотез ---')
    print('Гипотеза 1 (спорт возвращает лучше концертов):')
    print(f"  concert return_rate={hyp_event['return_rate_concert']:.2%}, "
          f"sport return_rate={hyp_event['return_rate_sport']:.2%}, "
          f"p-value={hyp_event['p_value']:.4f}")

    print('Гипотеза 2 (крупные регионы возвращают чаще):')
    print(f"  top10 return_rate={hyp_region['return_rate_top10']:.2%}, "
          f"rest return_rate={hyp_region['return_rate_rest']:.2%}, "
          f"p-value={hyp_region['p_value']:.4g}, "
          f"corr_spearman={hyp_region['corr_spearman']:.2f}")

    print('\nДень недели первой покупки (chi2-тест):')
    print(f"  chi2={weekday['chi2']:.2f}, p-value={weekday['p_value']:.4f}")
    print(f"  разброс return_rate по дням: "
          f"{weekday['stats']['return_rate'].max() - weekday['stats']['return_rate'].min():.2%}")

    print('\n--- Корреляционный анализ (phi_k с total_orders) ---')
    print(correlation['phik_matrix']['total_orders'].sort_values(ascending=False))

    print('\n--- Рекомендации ---')
    print('1. Развивать сегменты выставок и театра — повышенная лояльность с первого заказа.')
    print('2. Усилить работу со спортом и ёлками — ниже среднего return_rate при заметном объёме.')
    print('3. Не опираться на размер региона как единственный критерий потенциала.')
    print('4. В модели прогноза возврата использовать только признаки первого заказа '
          '(регион, тип мероприятия, устройство, день недели) — средний чек и интервал '
          'между покупками известны только постфактум (data leakage).')
    print('=' * 70)


def main():
    processor = DataProcessor()
    results = processor.run()
    print_report(results)


if __name__ == '__main__':
    main()
