import yaml
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, body, to_email, from_email, smtp_server, smtp_port, password):
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
    try:
        # 这里是你的神经网络训练代码
        # model.fit(X_train, y_train)
        print("Training started")
        # 假设训练代码
        for i in range(10000000):
            pass  # 代表训练过程

        # 训练完成后发送邮件通知
        send_email(
            subject="Training Complete",
            body="The training of your neural network has completed successfully.",
            to_email="recipient@example.com",
            from_email="your_email@example.com",
            smtp_server="smtp.example.com",
            smtp_port=587,
            login="your_email@example.com",
            password="your_password"
        )
    except Exception as e:
        # 捕获任何错误并发送错误通知邮件
        send_email(
            subject="Training Error",
            body=f"An error occurred during the training process: {e}",
            to_email="recipient@example.com",
            from_email="your_email@example.com",
            smtp_server="smtp.example.com",
            smtp_port=587,
            login="your_email@example.com",
            password="your_password"
        )

if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    subject='Test'
    body='No Reply'
    smtp_server=config['email']['server']
    smtp_port=config['email']['port']
    from_email=config['email']['from']
    to_email=config['email']['to']
    password=config['email']['password']
    
    send_email(subject, body, to_email, from_email, smtp_server, smtp_port, password)
    
