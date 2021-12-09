import streamlit as st
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import scipy
# %matplotlib inline
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, PolynomialFeatures,OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

# import pandas_profiling as pp
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, classification_report, confusion_matrix
# from streamlit.stats import StatsHandler
from xgboost import XGBRFRegressor

# from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from pmdarima import auto_arima
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

data = pd.read_csv('avocado.csv')
# GUI
st.title('Data Science Project')
st.write('## Hass Avocado Price Prediction')
uploaded_file = st.file_uploader('Choose a file', type = ['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data.to_csv('avocado_new.csv', index = False)
data['Date'] = pd.to_datetime(data['Date'])
data = data.drop("Unnamed: 0", axis = 1)

data['month'] = data['Date'].dt.month
data['revenue'] = data['AveragePrice'].multiply(data['Total Volume'])

# Thêm mùa vụ vào dữ liệu
def seasonal_us(x):
    if x >= 4 and x <= 6:
        return 0 # mùa xuân
    elif x >= 7 and x <= 9:
        return 1 # mùa hè
    elif x >= 10 and x <= 12:
        return 2 # mùa thu
    else:
        return 3 # mùa đông
data['seasonal'] = data['month'].map(seasonal_us)

# GUI
menu = ['Business Objective', 'Data Understanding', 'Build Project', 'Price Prediction']
choice = st.sidebar.selectbox('Menu',menu)
if choice == 'Business Objective':
    st.subheader('Business Objective')
    st.write("""
    - Bơ “Hass”, một công ty có trụ sở tại Mexico, chuyên sản xuất nhiều loại quả bơ được bán ở Mỹ. Họ đã rất thành công trong những năm gần đây và muốn mở rộng. Vì vậy, họ muốn xây dựng mô hình hợp lý để dự đoán giá trung bình của bơ “Hass” ở Mỹ nhằm xem xét việc mở rộng các loại trang trại Bơ đang có cho việc trồng bơ ở các vùng khác.
    - Hiện tại: Công ty kinh doanh quả bơ ở rất nhiều vùng của nước Mỹ với 2 loại bơ là bơ thường và bơ hữu cơ, được đóng gói theo nhiều quy chuẩn (Small/Large/XLarge Bags), và có 3 PLU (Product Look Up) khác nhau (4046, 4225, 4770). Nhưng họ chưa có mô hình để dự đoán giá bơ cho việc mở rộng.
    """)
    st.write("""
    ###### => Mục tiêu/ Vấn đề: Xây dựng mô hình dự đoán giá trung bình của bơ “Hass” ở Mỹ.
    ###### => Xem xét việc mở rộng sản xuất, kinh doanh.
    """)
    st.image('Avocado.jpeg')
    st.write("""
    ##### Mùa bơ
    Thời tiết nước Mỹ đa dạng với 4 mùa.
    - Mùa xuân ấm áp kéo dài từ tháng 3 - tháng 6.
    - Mùa hè ở Mỹ thì nắng nóng và bắt đầu từ tháng 6 – kết thúc đến tháng 9.
    - Mùa thu mát mẻ, và từ tháng 9 – tháng 12
    - Mùa đông tại Mỹ thì khá lạnh, và kéo dài từ tháng 12 – tháng 3
    - Summer: June - August
    - Fall: September - November
    - Winter: December - February
    - Spring: March - May 
    """)
    st.write("""
    #### Part 1: Data Understanding
    - Xem xét tìm hiểu dữ liệu, phân tích các biến
    - Trực quan hóa một vài thuộc tính của dữ liệu
    """)
    st.write("""
    #### Part 2: Build Project
    - Thực hiện chạy các thuật toán Regression dự báo cho giá bơ
    - Thực hiện các kiểm định MAE, MSE, R2
    """)
    st.write("""
    #### Part 3: Price Prediction
    - ExponentialSmoothing - Time Series Algorithm
    - Facebook Prophet - Time Series Algorithm
    """)
elif choice == 'Data Understanding':
    numbers = [f for f in data.columns if data.dtypes[f]!='object']
    objects = [f for f in data.columns if data.dtypes[f]=='object']
    st.subheader('Data Understanding')
    st.write("""
    ###### Head and Tail data:
    """)
    st.dataframe(data.head(3))
    st.dataframe(data.tail(3))
    st.write("""
    ###### Kiểm tra dữ liệu có thuộc tính Object
    """)
    i=1
    for obj in objects:
        st.write(i, '/', obj, '\t', len(data[obj].unique()), ':', data[obj].unique())
        i+=1
    st.write("""###### Kiểm tra sự tương quan của dữ liệu""")
    st.dataframe(data.corr())

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("""###### So sánh tương tương quan biểu đồ phân phối giá bơ Oganic và Conventional""")
    for i in ['AveragePrice']:
        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        sns.histplot(data=data, x=data[i], kde=True, hue='type')
        plt.subplot(1,2,2)
        sns.boxplot(data=data, y=data[i], x='type')
        plt.show()
    st.pyplot()
    st.write('=> Qua biểu đồ cho thấy rằng giá bán các loại bơ ở các bang có phân phối tương đương phân phối chuẩn, giá bán bơ Organic cao hơn gấp rưỡi so với bơ Conventional, giá bán bơ Organic chứa nhiều outlier hơn so với bơ thường')
    st.write("""###### So sánh giá bơ Oganic và Conventional theo Region""")
    for i in ['AveragePrice']:
        plt.figure(figsize=(20,5))
        sns.boxplot(data=data, y=data[i], x='region',hue='type')
        plt.xticks(rotation=60, ha="right")
        plt.show()
    st.pyplot()
    st.write('=> biểu đồ này cho thấy bơ thường có giá thấp hơn hẳn bơ Organic ở tất cả các bang')
    st.write("""###### Doanh thu bán hàng bơ Oganic và Conventional theo Region""")
    plt.figure(figsize=[20,5])
    df_ = data[data['region']!='TotalUS'].groupby(['region','type']).sum().reset_index()[['region','type','revenue']].sort_values('revenue',ascending=False)
    g = sns.barplot(x='region',y='revenue',hue='type',data=df_)
    for p in g.patches:
        _x = p.get_x() + p.get_width()/2
        _y = p.get_y() + p.get_height()/0.97
        value = int(p.get_height())
        g.text(_x, _y, round(value/1e9,2), ha='center',va='bottom',size=9,rotation=90)
    plt.title('Total Revenue '+str(round(sum(df_['revenue'])/1e9,2))+' Bill USD',size=20)
    plt.ticklabel_format(axis='y',scilimits=(9,9))
    plt.grid(axis='y')
    plt.xticks(rotation=60, ha="right")
    plt.show()
    st.pyplot()
    st.write('=> Doanh thu bơ thường cao hơn rất nhiều so với bơ Organic, chứng tỏ giá bán Organic cao mà sản lượng không nhiều')
elif choice == 'Build Project':
    st.subheader('Build Project')
    with st.form(key='Lựa chọn tham số mô hình'):
        # model
        train_ratio = st.slider('Tỉ lệ tập train', 0.7, 0.9, 0.8)
        # algorithm = st.radio('Thuật toán', options=[RandomForestRegressor(), LinearRegression(), DecisionTreeRegressor()])
        # scaler = st.radio('Phương pháp Scaler', options=[RobustScaler(),StandardScaler()])
        submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        def onehot_encoder(data_frame, list_of_field_encoder):
            for i in list_of_field_encoder:
                onehot_encoder = OneHotEncoder()
                onehot_encoder.fit(data_frame[[i]])
                encoder = onehot_encoder.transform(data_frame[[i]]).toarray()
                dfOneHot = pd.DataFrame(encoder, columns=onehot_encoder.categories_[0])
                data_frame = pd.concat([data_frame,dfOneHot], axis=1).drop([i],axis=1)
            return data_frame
        # Xây dự hàm pipe thực hiện các công việc tương tự
        def SS_PF_LR(df_X,df_Y,Train_Size,Scaler,Polynomial,Modal):
            import time
            t = time.time()
            if Train_Size == 1:
                X_train, X_test, Y_train, Y_test = df_X, df_X, df_Y, df_Y
            else:
                X_train, X_test, Y_train, Y_test = train_test_split(df_X,df_Y,train_size=Train_Size)
            Input = [('Scaler',Scaler),('Polynomial',Polynomial),('Modal',Modal)]
            pipe = Pipeline(Input)
            pipe.fit(X_train,Y_train)
            Y_Pipe = pipe.predict(X_test)
            st.write('R2_Score - Train:',pipe.score(X_train,Y_train))
            st.write('R2_Score - Test:',pipe.score(X_test,Y_test))
            st.write('MAE', mean_absolute_error(Y_test,Y_Pipe))
            st.write('MSE', mean_squared_error(Y_test,Y_Pipe))
            st.write('Time(s)', time.time()-t)
            # st.write('r2_score', r2_score(Y_test,Y_Pipe))
            return X_train, X_test, Y_train, Y_test, pipe, Y_Pipe
        # Select X, Y data
        X = data[['type','year','region','month','seasonal','Total Volume','Total Bags']] # inputs
        y = data['AveragePrice'] # output
        X_lb = onehot_encoder(X.interpolate(),['type','region','year'])
        st.write('Head of data after OneHotEncoder')
        st.dataframe(X_lb.head(5))
        st.write('Starting Regression...')
        col1, col2 = st.columns([5,5])
        with col1:
            st.write('#### Kết quả thuật toán LinearRegression')
            SS_PF_LR(X_lb,y,train_ratio,RobustScaler(),PolynomialFeatures(degree=1),LinearRegression())
        with col2:
            st.write('#### Kết quả thuật toán RandomForestRegressor')
            SS_PF_LR(X_lb,y,train_ratio,RobustScaler(),PolynomialFeatures(degree=1),RandomForestRegressor())
        col3, col4 = st.columns([5,5])
        with col3:
            st.write('#### Kết quả thuật toán DecisionTreeRegressor')
            SS_PF_LR(X_lb,y,train_ratio,RobustScaler(),PolynomialFeatures(degree=1),DecisionTreeRegressor())
        with col4:
            st.write('#### Kết quả thuật toán XGBRFRegressor')
            SS_PF_LR(X_lb,y,train_ratio,RobustScaler(),PolynomialFeatures(degree=1),XGBRFRegressor())
        st.write('Done!')
        st.write("""
        ### Nhận Xét:
        - Thuật toán RandomForestRegressor cho kết quả dự báo tập train và test cao nhất, LinearRegression cho kết quả không cao nhưng chi phí thực hiện là thấp nhất
        """)
elif choice == 'Price Prediction':
    st.subheader('Price Prediction')
    # lấy list region và list type avocado
    lst_region = data['region'].unique().tolist()
    lst_type = data['type'].unique().tolist()
    with st.form(key='Lựa chọn loại bơ và vùng muốn dự báo giá:'):
        # predic = st.radio('Lựa chọn dự báo giá hoặc sản lượng:', options=['AveragePrice','Total Volume'])
        type_predic = st.multiselect('Lựa chọn loại bơ bạn muốn dự báo giá:', lst_type, default=['organic'])
        region_predic = st.multiselect('Lựa chọn vùng bạn muốn dự báo giá:', lst_region, default=['California'])
        train_ratio = st.slider('Tỉ lệ tập train timeseri:', 0.7, 0.9, 0.8)
        values = st.slider('Lựa chọn khoảng chu kỳ để tìm chu kỳ mùa vụ cho mô hình ExponentialSmoothing:', 0, 365, (45, 55))
        select_predict = st.slider('Thời gian bạn muốn dự báo (năm):', 1, 5, 1)
        # end_date = st.date_input('End date', tomorrow)
        submit_button = st.form_submit_button(label='Submit')
    if submit_button:
# Dự AveragePrice
        st.write('## Dự báo AveragePrice')
        data_time = data[(data['type'].isin(type_predic))&(data['region'].isin(region_predic))][['Date','AveragePrice']].groupby('Date').mean().sort_index()
        st.write('Head of timeseri data')
        st.dataframe(data_time.head(3))
        st.write('Tail of timeseri data')
        st.dataframe(data_time.tail(3))
        st.write("Trend và tính mùa vụ của giá trung bình bơ %s ở bang %s" % (type_predic,region_predic))
        result = seasonal_decompose(data_time['AveragePrice'], model='multiplicative')
        plt.figure(figsize=(15,4))
        plt.plot(result.trend)
        plt.title("Trend AveragePrice")
        st.pyplot()
        plt.figure(figsize=(15,4))
        plt.plot(result.seasonal)
        plt.title("Seasonal AveragePrice")
        st.pyplot()
        # dự báo
        st.write('#### ExponentialSmoothing')
        train, test = data_time.iloc[:int(round(train_ratio*data_time.shape[0],0)), 0], data_time.iloc[int(round(train_ratio*data_time.shape[0],0)):, 0] # 80% train
        st.write('Lựa chọn tham số cho mô hình')
        seasonal_periods=[]
        MSE_test=[]
        MAE_test=[]
        MSE_train=[]
        MAE_train=[]
        for i in range(values[0],values[1]):
            model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=i).fit()
            pred = model.predict(start=test.index[0], end=test.index[-1])
            pred_train = model.predict(start=train.index[0], end=train.index[-1])
            seasonal_periods.append(i)
            MSE_test.append(mean_squared_error(test,pred))
            MAE_test.append(mean_absolute_error(test,pred))
            MSE_train.append(mean_squared_error(train,pred_train))
            MAE_train.append(mean_absolute_error(train,pred_train))
        Result = pd.DataFrame({'Seasonal': seasonal_periods,'MSE test': MSE_test,'MAE test': MAE_test,'MSE train': MSE_train,'MAE train': MAE_train}, columns=['Seasonal','MSE test','MAE test','MSE train','MAE train'])
        ## bảng kết quả và lựa chọn mô hình tối ưu
        st.dataframe(Result.sort_values('MSE test').head(10))
        seasonal_final = Result[Result['MSE test']==min(Result['MSE test'])].iloc[0,0]
        st.write('Lựa chọn mô hình có Seasonal bằng %.f cho mô hình ExponentialSmoothing'%(seasonal_final))
        st.write('Thực hiện dự báo bằng mô hình ExponentialSmoothing')
        model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=seasonal_final).fit()
        pred = model.predict(start=test.index[0], end=test.index[-1])
        # fit model cho 1 năm
        model = ExponentialSmoothing(data_time, seasonal='mul', seasonal_periods=seasonal_final).fit()
        import datetime
        s = data_time.index.max() # cần tạo biến
        e = data_time.index.max() + datetime.timedelta(days=365*select_predict)
        pred_next_one_year = model.predict(start= s, end=e)
        st.write('Dự báo cho %.f năm tới:'%(select_predict))
        st.dataframe(pred_next_one_year)
        st.write('Biểu đồ so sánh và đánh giá:')
        x = pd.Series(pred_next_one_year)
        plt.figure(figsize=(15,6))
        plt.plot(train.index, train, label='Train')
        plt.plot(test.index, test, label='Test')
        plt.plot(pred.index, pred, label='Holt-Winters')
        plt.plot(x.index, x.values, label='Next-One-Year')
        plt.title("ExponentialSmoothing Model")
        plt.legend(loc='best')
        st.pyplot()

        st.write('#### FaceBook Prophet')
        model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
        model.fit(data_time.reset_index().rename(columns={'Date':'ds','AveragePrice':'y'}))
        # 52 weeks
        future = model.make_future_dataframe(periods=52*select_predict, freq='W')
        forecast=model.predict(future)
        st.write('Bảng kết quả cho %.f năm tới:'%(select_predict))
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']])
        st.write('Trực quan hóa khoảng tin cậy FBProphet')
        plt.figure(figsize=(15,8))
        fig = model.plot(forecast)
        fig.show()
        a = add_changepoints_to_plot(fig.gca(), model, forecast)
        plt.plot(data_time, label='Average Price', color='green')
        plt.title('FaceBook Prophet')
        st.pyplot()
        st.write('Xem xét yếu tố mùa vụ và trend của FBProphet đưa ra:')
        fig1 = model.plot_components(forecast)
        st.pyplot(fig1)
# Dự báo sản lượng
        st.write('## Dự báo Total Volume')
        data_time = data[(data['type'].isin(type_predic))&(data['region'].isin(region_predic))][['Date','Total Volume']].groupby('Date').sum().sort_index()
        st.write('Head of timeseri data')
        st.dataframe(data_time.head(3))
        st.write('Tail of timeseri data')
        st.dataframe(data_time.tail(3))
        st.write("Trend và tính mùa vụ của sản lượng bơ %s ở bang %s" % (type_predic,region_predic))
        result = seasonal_decompose(data_time['Total Volume'], model='multiplicative')
        plt.figure(figsize=(15,4))
        plt.plot(result.trend)
        plt.title("Trend Total Volume")
        st.pyplot()
        plt.figure(figsize=(15,4))
        plt.plot(result.seasonal)
        plt.title("Seasonal Total Volume")
        st.pyplot()
        # dự báo
        st.write('#### ExponentialSmoothing')
        train, test = data_time.iloc[:int(round(train_ratio*data_time.shape[0],0)), 0], data_time.iloc[int(round(train_ratio*data_time.shape[0],0)):, 0] # 80% train
        st.write('Lựa chọn tham số cho mô hình')
        seasonal_periods=[]
        MSE_test=[]
        MAE_test=[]
        MSE_train=[]
        MAE_train=[]
        for i in range(values[0],values[1]):
            model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=i).fit()
            pred = model.predict(start=test.index[0], end=test.index[-1])
            pred_train = model.predict(start=train.index[0], end=train.index[-1])
            seasonal_periods.append(i)
            MSE_test.append(mean_squared_error(test,pred))
            MAE_test.append(mean_absolute_error(test,pred))
            MSE_train.append(mean_squared_error(train,pred_train))
            MAE_train.append(mean_absolute_error(train,pred_train))
        Result = pd.DataFrame({'Seasonal': seasonal_periods,'MSE test': MSE_test,'MAE test': MAE_test,'MSE train': MSE_train,'MAE train': MAE_train}, columns=['Seasonal','MSE test','MAE test','MSE train','MAE train'])
        ## bảng kết quả và lựa chọn mô hình tối ưu
        st.dataframe(Result.sort_values('MSE test').head(10))
        seasonal_final = Result[Result['MSE test']==min(Result['MSE test'])].iloc[0,0]
        st.write('Lựa chọn mô hình có Seasonal bằng %.f cho mô hình ExponentialSmoothing'%(seasonal_final))
        st.write('Thực hiện dự báo bằng mô hình ExponentialSmoothing')
        model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=seasonal_final).fit()
        pred = model.predict(start=test.index[0], end=test.index[-1])
        # fit model cho 1 năm
        model = ExponentialSmoothing(data_time, seasonal='mul', seasonal_periods=seasonal_final).fit()
        import datetime
        s = data_time.index.max() # cần tạo biến
        e = data_time.index.max() + datetime.timedelta(days=365*select_predict)
        pred_next_one_year = model.predict(start= s, end=e)
        st.write('Dự báo cho %.f năm tới:'%(select_predict))
        st.dataframe(pred_next_one_year)
        st.write('Biểu đồ so sánh và đánh giá:')
        x = pd.Series(pred_next_one_year)
        plt.figure(figsize=(15,6))
        plt.plot(train.index, train, label='Train')
        plt.plot(test.index, test, label='Test')
        plt.plot(pred.index, pred, label='Holt-Winters')
        plt.plot(x.index, x.values, label='Next-One-Year')
        plt.title("ExponentialSmoothing Model")
        plt.legend(loc='best')
        st.pyplot()

        st.write('#### FaceBook Prophet')
        model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
        model.fit(data_time.reset_index().rename(columns={'Date':'ds','Total Volume':'y'}))
        # 52 weeks
        future = model.make_future_dataframe(periods = 52*select_predict, freq='W')
        forecast=model.predict(future)
        st.write('Bảng kết quả dự báo:')
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']])
        st.write('Trực quan hóa khoảng tin cậy FBProphet')
        plt.figure(figsize=(15,8))
        fig = model.plot(forecast)
        fig.show()
        a = add_changepoints_to_plot(fig.gca(), model, forecast)
        plt.plot(data_time, label='Total Volume', color='green')
        plt.title('FaceBook Prophet')
        st.pyplot()
        st.write('Xem xét yếu tố mùa vụ và tren của FBProphet đưa ra:')
        fig1 = model.plot_components(forecast)
        st.pyplot(fig1)

        st.write("""
        ### Nhận xét: Từ kết quả dự báo cho thấy đồ thị dự báo đi khá sát so với thực tế, sản lượng bơ nhiều vào đầu năm dẫn tới giá bơ sẽ thấp hơn vào đầu năm, về cuối năm khi bước vào mùa đông thì sản lượng bơ giảm > khan hiếm bơ trên thị trường nên đẩy giá bơ lên cao.
        """)
        st.write("""
        Kết luận tư vấn cho nhà quản trị: chưa nên mở rộng sản lượng bơ Organic tại bang California với điều kiện thị trường hiện tại, nhưng cần đẩy giá bơ tăng hàng năm để theo tỉ lệ trượt giá của nền kinh tế. Ngoài ra cần xem xét các yếu tố ảnh hưởng khác trong nền kinh tế vĩ mô và vi mô, những yếu tố đó thì không được thể hiện trong dữ liệu hiện tại, nhưng chỉ là đột biến về xu thế giá và sản lượng mô hình dự báo cho thấy kết quả khá sát với thực tế
        """)