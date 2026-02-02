from imap_tools import MailBox, AND

# Replace with your Outlook email and app password
EMAIL = "baburpa@outlook.com"
PASSWORD = "Posi@143"

# Connect to Outlook IMAP
with MailBox('outlook.office365.com').login(EMAIL, PASSWORD) as mailbox:
    # Fetch all emails from INBOX
    for msg in mailbox.fetch(AND(all=True)):
        print("From:", msg.from_)
        print("Subject:", msg.subject)
        print("Date:", msg.date)
        print("Body:", msg.text or msg.html)
        print("-" * 50)