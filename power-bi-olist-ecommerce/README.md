# E-commerce Dashboard: Olist (Power BI)

Interactive Power BI report built on the public [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce): about 100k orders, 2016–2018.

**Pages:** Sales · Logistics · Payments · Feedback

- Star-schema model: orders, order items, payments, reviews, customers, products, sellers, plus a calendar table
- DAX measures for leads, delivered sales and revenue, with a metric selector (field parameter) and a sales funnel
- Delivery performance, payment methods and review scores by period, category and region

**Open:** `olist_ecommerce_dashboard.pbix` in Power BI Desktop. The data was imported from the dataset's CSV files. No database connection is used.
