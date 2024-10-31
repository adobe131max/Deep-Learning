import yaml
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def sendEmail(subject, body, path):
    def decorator(func):
        def wrapper(*args, **kwargs):
            ans = func(*args, **kwargs)
            send_email(subject, body, path)
            return ans
        return wrapper
    return decorator


def send_email(subject, body, path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        
    smtp_server=config['email']['server']
    smtp_port=config['email']['port']
    from_email=config['email']['from']
    to_email=config['email']['to']
    password=config['email']['password']
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

def train_model():
    config_path = './config.yaml'
    try:
        # 这里是你的神经网络训练代码
        # model.fit(X_train, y_train)
        print("Training started")
        # 假设训练代码
        for i in range(10000000):
            pass  # 代表训练过程

        # 训练完成后发送邮件通知
        send_email('Training Complete', 'The training of your neural network has completed successfully.', config_path)
    except Exception as e:
        # 捕获任何错误并发送错误通知邮件
        send_email('Training Error', f'An error occurred during the training process: {e}', config_path)

if __name__ == '__main__':
    config_path = './config.yaml'
    subject='Test'
    body='No Reply'
    send_email(subject, body, config_path)
    
