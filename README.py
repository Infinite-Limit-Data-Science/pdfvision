import pandas as pd
from datetime import datetime
from dateutil.parser import parse

from .scripts.llama32 import llama32
from .scripts.Pixtral import pixtral

import json
from .functions_prompt import system_prompt
from .scripts.ExchangeInterfaceV6 import Emails
from .scripts.Utils import (
    get_pdf_summary,
    extract_all_strings_regex,
    validateVoucher,
    createDocumentBundle,
    get_doc_summary,
    process_doc,
    pymuLLM,
    extract_invoice_from_pdf_attachment,
    triagePeopleSoftInvalid,
    splunkit,
)
from .scripts.dbpandas import Insertlogstodb
from .prompts import (
    pdf_prompt_text,
    image_prompt_text,
    step1_prompt,
    step2_prompt,
    move_to_rush_prompt,
    check_invoice_prompt,
    check_again_prompt,
    move_to_AiEmily_folder_prompt,
    move_to_non_invoice_prompt,
    get_and_process_attachments_prompt,
    create_draft_response_prompt,
    move_to_business_folder_prompt,
    body_prompt_text,
    CDR_lineMissing_prompt,
    CDR_unableToProcessMaxTokens_prompt,
    CDR_UnreadableAttachment_prompt,
    CDR_VendorUpdate_prompt,
    CDR_quantitymultiplelines_prompt,
    CDR_duplicate_invoice_prompt,
    CDR_ConnectionError_prompt,
)

import sys
from .scripts.gwlogging import init_logger

import schedule
import time
import base64
import logging
from logging.handlers import RotatingFileHandler
import os
import re

# from prompts import pdf_prompt_text

exchangeMailbox = Emails()


class CurrentWorkingEmail:
    def __init__(self, data=None):
        if data is None:
            self.id = None
            self.created_date_time = None
            self.last_modified_date_time = None
            self.change_key = None
            self.categories = None
            self.received_date_time = None
            self.sent_date_time = None
            self.has_attachments = None
            self.internet_message_id = None
            self.subject = None
            self.body_preview = None
            self.importance = None
            self.parent_folder_id = None
            self.conversation_id = None
            self.conversation_index = None
            self.is_delivery_receipt_requested = None
            self.is_read_receipt_requested = None
            self.is_read = None
            self.is_draft = None
            self.web_link = None
            self.inference_classification = None
            self.body = None
            self.sender = None
            self.from_ = None
            self.to_recipients = None
            self.cc_recipients = None
            self.bcc_recipients = None
            self.reply_to = None
            self.flag = None
        else:
            self.id = data["value"][0].get("id")
            self.created_date_time = data["value"][0].get("createdDateTime")
            self.last_modified_date_time = data["value"][0].get("lastModifiedDateTime")
            self.change_key = data["value"][0].get("changeKey")
            self.categories = data["value"][0].get("categories")
            self.received_date_time = data["value"][0].get("receivedDateTime")
            self.sent_date_time = data["value"][0].get("sentDateTime")
            self.has_attachments = data["value"][0].get("hasAttachments")
            self.internet_message_id = data["value"][0].get("internetMessageId")
            self.subject = data["value"][0].get("subject")
            self.body_preview = data["value"][0].get("bodyPreview")
            self.importance = data["value"][0].get("importance")
            self.parent_folder_id = data["value"][0].get("parentFolderId")
            self.conversation_id = data["value"][0].get("conversationId")
            self.conversation_index = data["value"][0].get("conversationIndex")
            self.is_delivery_receipt_requested = data["value"][0].get(
                "isDeliveryReceiptRequested"
            )
            self.is_read_receipt_requested = data["value"][0].get(
                "isReadReceiptRequested"
            )
            self.is_read = data["value"][0].get("isRead")
            self.is_draft = data["value"][0].get("isDraft")
            self.web_link = data["value"][0].get("webLink")
            self.inference_classification = data["value"][0].get(
                "inferenceClassification"
            )
            self.body = data["value"][0].get("body")
            self.sender = data["value"][0].get("sender")
            self.from_ = data["value"][0].get("from")
            self.to_recipients = data["value"][0].get("toRecipients")
            self.cc_recipients = data["value"][0].get("ccRecipients")
            self.bcc_recipients = data["value"][0].get("bccRecipients")
            self.reply_to = data["value"][0].get("replyTo")
            self.flag = data["value"][0].get("flag")
            self.attachments = []
            self.voucher = []
            self.mime_message = None

    def assign(self, data):
        self.__init__(data)

    def assignAttachment(self, data):
        self.attachments.append(Attachment(data))  # self.attachments = Attachment(data)

    def assignVoucher(self, data):
        self.voucher.append(Voucher(data))

    def assignMimeMessage(self, data):
        self.mime_message = data

    def __str__(self):
        return (
            f"CurrentWorkingEmail(id={self.id}, subject={self.subject}, "
            f"sender={self.sender}, recipients={self.to_recipients})"
        )


class Attachment:
    def __init__(self, data=None):
        if data is None:
            self.id = None
            self.last_modified_date_time = None
            self.name = None
            self.content_type = None
            self.size = None
            self.is_inline = None
            self.content_id = None
            self.content_location = None
            self.content_bytes = None
        else:
            self.id = data["id"]
            self.last_modified_date_time = data["lastModifiedDateTime"]
            self.name = data["name"]
            self.content_type = data["contentType"]
            self.size = data["size"]
            self.is_inline = data["isInline"]
            self.content_id = data["contentId"]
            self.content_location = data["contentLocation"]
            self.content_bytes = data["contentBytes"]
            # self.voucher = []

    def assign(self, data):
        self.__init__(data)

    # def assignVoucher(self, data):
    #     self.voucher.append(Voucher(data))

    def __str__(self):
        return f"Attachment(id={self.id}, name={self.name})"


class Voucher:
    def __init__(self, data=None):
        if data is not None:
            self.invoice_date = data.get("invoice_date")
            self.invoice_number = data.get("invoice_number")
            self.gross_invoice_amount = data.get("gross_invoice_amount")
            self.invoice_tax = data.get("invoice_tax")
            self.invoice_freight = data.get("invoice_freight")
            self.po_number = data.get("po_number")
            self.po_line_number = data.get("po_line_number")
            self.po_line_amount = data.get("po_line_amount")
            self.invoice_description = data.get("invoice_description")
            self.isvalid = data.get("isvalid")
            self.PO_BALANCE = data.get("PO_BALANCE")
            self.isDuplicateInvoice = data.get("isDuplicateInvoice")
            self.ap_business_unit = data.get("ap_business_unit")
            self.po_business_unit = data.get("po_business_unit")
            self.vendor_id = data.get("vendor_id")
            self.reason = data.get("Reason")

    def assign(self, data):
        self.__init__(data)

    def __str__(self):
        return (
            f"Voucher(invoice_date={self.invoice_date}, "
            f"invoice_number={self.invoice_number})"
        )


def getMessageList(count, folder):
    """Return a list of emails matching the specified ask. Default value for count is 10 and default folder is Inbox"""
    msgList = exchangeMailbox.listEmails(count, folder)

    # with open('message.txt', 'w') as f:
    #      json.dump(msgList, f)

    if msgList is not None:
        for item in msgList:
            print(item)
        # print(msg)
        return msgList
    return None


def getEmailAttachments(id):
    """Returns the attachments of the specific email with the specified id"""
    attachments = exchangeMailbox.getAttachments(id).json()

    # with open('attachment.txt', 'w') as f:
    #      json.dump(attachments, f)

    for item in attachments:
        print(item)

    return attachments


def getMIMEMessage(id):
    """Returns the MIME message of the specific email with the specified id"""
    msg = exchangeMailbox.getMIMEMessage(id)

    # for item in msg:
    #     print(item)

    return msg


def processAttachments(attachments):
    """Process the attachments files contained in the attachments list"""
    for attachment in attachments:
        print(f"attachment id {attachment.id}")
        print(f"attachment.name {attachment.name}")
        print(f"attachment.content_type {attachment.content_type}")
        print(f"attachment.size {attachment.size}")
        print(f"attachment.is_inline {attachment.is_inline}")
        print(f"attachment.content_id {attachment.content_id}")
        print(f"attachment.content_location{attachment.content_location}")
        # print(f'attachment.content_bytes {attachment.content_bytes}')

        if attachment.content_type == "text/plain":
            print(attachment.content_bytes.decode("utf-8"))
        elif attachment.content_type == "application/pdf":
            pdfText = get_pdf_summary(attachment.content_bytes)
            print(pdfText)
        elif attachment.content_type == "image/png":
            print(attachment.content_bytes)
        else:
            print("unknown content type")
    return True


def moveMessage(id, folderName):
    """Moves the specific email with the specified id to the specified folder"""
    # agent.PS_BC_AI_RPT_LOG_DF.at[0,'BC_REMARKS'] = "Message moved to " + folderName
    return exchangeMailbox.moveMessage(id, folderName)


def createReplyMessage(id, body):
    """Creates a reply to the specific email with the specified id"""
    # agent.PS_BC_AI_RPT_LOG_DF.at[0,'ERROR_MSG_TXT'] = "Draft Message reply created " + to
    return exchangeMailbox.createReplyMessage(id, body)


def createNewMessage(subject, body, toRecipients, ccRecipients=None, bccRecipients=None, attachments=None):
    return exchangeMailbox.createNewMessage(subject, body, toRecipients, ccRecipients, bccRecipients, attachments)


def updateEmailDisposition(incoming_df):
    """function will update the outcome of the email handler with remarks. All the required data will be passed in the dataframe PS_BC_AI_RPT_LOG_DF"""
    print("\ninside updateEmailDisposition routine\n")

    # send_df = pd.read_json(incoming_df)
    # print(send_df)
    # Insertlogstodb(send_df)

    Insertlogstodb(incoming_df)  # update the email disposition commented out on 06/02/2025 because it was not working in the image
    return True


def checkImageforInvoice(content_type, content_bytes):
    """function will check for image in the email and determine if it is an invoice or not."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Determine if the image is an invoice or not. Invoices typically include the word 'Invoice' Answer with either Yes or No.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{content_type};base64,{content_bytes}"},
                },
            ],
        }
    ]

    result = llama32(messages)
    print(f"llama return {result}")
    splunkit(f"\nLlama: {result}", "info")

    result = pixtral(messages)
    print(f"pixtral return {result}")
    splunkit(f"\nPixtral: {result}", "info")

    if "Yes" in result:
        return True
    if "No" in result:
        return False
    return True


def classifyAttachment(text):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Given the text extracted from an attachment, determine if the attachment is an invoice or not. "
                        "Invoices typically include the word 'Invoice'. Answer with either 'Yes' or 'No'. "
                        f"Attachment text: {text}"
                    ),
                },
            ],
        }
    ]

    result = llama32(messages)
    if result == "Yes":
        return True
    if result == "max_new_token_error":
        return "max_new_token_error"
    return False


def validateVoucherOutcome(workingEmail):
    """Function to analyse voucher outcome."""
    for voucher in workingEmail.voucher:
        print(vars(voucher))
    return True


class Agent:
    """This is the agent class that invokes tools and functions to work on the email and its attachements."""

    def __init__(self, system_prompt=None):
        current_directory = os.getcwd()
        outputdirectory = os.path.join("./", "data", "logs")
        os.makedirs(outputdirectory, exist_ok=True)

        # self.logfilename = f"{outputdirectory}/agent-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
        self.logfilename = f"{outputdirectory}/AgenticAi-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

        handler = RotatingFileHandler(
            self.logfilename,
            maxBytes=100 * 1024 * 1024,
            backupCount=1,
            encoding="utf-8",
        )
        # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename=self.logfilename,filemode="w")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        # Create a logger
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(handler)

        # self.logger = init_logger()
        try:
            self.logger.info(f"\nStarting the Email Agent at : {datetime.now()}")
            splunkit(f"\nStarting the Email Agent at : {datetime.now()}", "info")

            self.workingEmail = CurrentWorkingEmail()
            self.MIMEMessage = []
            self.system_prompt = system_prompt
            self.messages = []
            self.bodyTxt = ""
            self.RTB_reason = ""

            if system_prompt:
                self.messages.append({"role": "system", "content": system_prompt})

                # Define the column names and data types for the log table for email dispositions
                # hosted in peopleSoft as a bolt on table. table name is PS_BC_AI_RPT_LOG
                columns = {
                    "BC_MSG_ID": "VARCHAR2 (254 Byte)",
                    "BC_EMAIL_SUBJTEXT": "VARCHAR2 (254 Byte)",
                    "EMAIL_DATETIME": "TIMESTAMP(6)",
                    "INVOICE_ID": "VARCHAR2 (30 Byte)",
                    "INVOICE_AMOUNT": "NUMBER (26,3)",
                    "PO_ID": "VARCHAR2 (10 Byte)",
                    "INVOICE_DT": "VARCHAR2 (10 Byte)",
                    "EXPORT_DATE": "DATE",
                    "ERROR_MSG_TXT": "VARCHAR2 (254 Byte)",
                    "BC_RUNTIME": "TIMESTAMP(6)",
                    "BC_REMARKS": "VARCHAR2 (254 Byte)",
                }

                # Create an empty DataFrame with the specified columns
                self.PS_BC_AI_RPT_LOG_DF = pd.DataFrame(columns=list(columns.keys()))

                # Set the data types for each column
                for col, dtype in columns.items():
                    self.PS_BC_AI_RPT_LOG_DF[col] = self.PS_BC_AI_RPT_LOG_DF[col].astype(dtype)

                self.PS_BC_AI_RPT_LOG_DF["BC_MSG_ID"] = "NULL"
                self.PS_BC_AI_RPT_LOG_DF["EMAIL_DATETIME"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.PS_BC_AI_RPT_LOG_DF["INVOICE_ID"] = 0
                self.PS_BC_AI_RPT_LOG_DF["INVOICE_AMOUNT"] = 0.00
                self.PS_BC_AI_RPT_LOG_DF["PO_ID"] = 0
                self.PS_BC_AI_RPT_LOG_DF["INVOICE_DT"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.PS_BC_AI_RPT_LOG_DF["EXPORT_DATE"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.PS_BC_AI_RPT_LOG_DF["ERROR_MSG_TXT"] = "Initializing the Email Agent. " + str(datetime.now())
                self.PS_BC_AI_RPT_LOG_DF["BC_RUNTIME"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.PS_BC_AI_RPT_LOG_DF["BC_REMARKS"] = "Initializing the Email Agent. " + str(datetime.now())
                self.PS_BC_AI_RPT_LOG_DF["BC_EMAIL_SUBJTEXT"] = " "

                print(self.PS_BC_AI_RPT_LOG_DF)

                self.logger.info("Initializing the Email Agent completed. " + str(datetime.now()))
                splunkit("Initializing the Email Agent completed. " + str(datetime.now()), "info")
                self.logger.info(f"PS_BC_AI_RPT_LOG_DF: {self.PS_BC_AI_RPT_LOG_DF}")
                splunkit(f"PS_BC_AI_RPT_LOG_DF: {self.PS_BC_AI_RPT_LOG_DF}", "info")
                # print(self.messages)

        except Exception as e:
            self.logger.error(f"AgentV2: Error in __init__: {e} ")
            splunkit(f"AgentV2: Error in __init__: {e} ", "error")

    def __call__(self, user_prompt_or_tool_result, is_tool_call=False):
        # if it's tool call result, use "ipython" instead of "user" for the role
        try:
            newVoucher = "DUMMY"
            self.messages.append(
                {"role": ("ipython" if is_tool_call else "user"), "content": user_prompt_or_tool_result}
            )

            result = self.llama()

            print(f"\nLlama returned: {result}.")

            if isinstance(result, str):
                checkFor500 = result.replace("(", "").split(",")[0].strip()
                if checkFor500 in ("500", "501", "502", "503"):
                    raise Exception(ConnectionError)
                return result

            if isinstance(result, dict):  # result is a dict only if it's a tool call spec
                try:
                    function_name = result["function_name"]
                    func = globals()[function_name]
                    parameters = result["parameters"]
                    result = func(**parameters)

                    if function_name == "getEmailAttachments":  # omitting the attachment from logging
                        self.logger.info(f"Tool calling call {function_name} with parameters {parameters}")
                        splunkit(f"Tool calling call {function_name} with parameters {parameters}", "info")
                    else:
                        self.logger.info(
                            f"Tool calling call {function_name} with parameters {parameters} returned: {result}"
                        )
                        splunkit(
                            f"Tool calling call {function_name} with parameters {parameters} returned: {result}",
                            "info",
                        )

                    print(f"\nTool calling returned: {result}")

                    if function_name == "getMessageList":
                        print("\nGet message list function called")
                        print(len(result["value"]))

                        if len(result["value"]) > 0:
                            self.workingEmail.assign(result)
                            self.bodyTxt = self.workingEmail.body["content"]

                            print(self.workingEmail)

                            # update the log table entries with email dispositions
                            self.PS_BC_AI_RPT_LOG_DF.at[0, "BC_MSG_ID"] = self.workingEmail.id
                            format_date_time = datetime.fromisoformat(self.workingEmail.received_date_time)
                            self.PS_BC_AI_RPT_LOG_DF.at[0, "EMAIL_DATETIME"] = format_date_time.strftime("%Y-%m-%d %H:%M:%S")
                            self.PS_BC_AI_RPT_LOG_DF["EXPORT_DATE"] = datetime.now().strftime("%Y-%m-%d")
                            self.PS_BC_AI_RPT_LOG_DF.at[0, "BC_EMAIL_SUBJTEXT"] = self.workingEmail.subject
                            self.PS_BC_AI_RPT_LOG_DF.at[0, "ERROR_MSG_TXT"] = "Step 1: getMessageList - Success"
                            self.PS_BC_AI_RPT_LOG_DF.at[0, "BC_REMARKS"] = "Successfuly read the email from the inbox"

                            self.logger.info(f"PS_BC_AI_RPT_LOG_DF: {self.PS_BC_AI_RPT_LOG_DF}")
                            splunkit(f"PS_BC_AI_RPT_LOG_DF: {self.PS_BC_AI_RPT_LOG_DF}", "info")
                            print(self.PS_BC_AI_RPT_LOG_DF)

                            self.messages.append(
                                {
                                    "role": "assistant",
                                    "content": "getMessage list successfully fetched the first email from Inbox. PS_BC_AI_RPT_LOG_DF is also updated.",
                                }
                            )
                            self.messages.append({"role": "assistant", "content": str(result)})  # keep track of response
                            self.messages.append(
                                {
                                    "role": "assistant",
                                    "content": f"you dont have to invoke getMessageList function again. From now onwards you can use the message id {self.workingEmail.id} to process any other tasks.",
                                }
                            )
                        else:
                            self.messages.append({"role": "assistant", "content": str(result)})
                            self.messages.append(
                                {"role": "assistant", "content": "There are no emails in the Inbox. Please check the Inbox."}
                            )
                            result = "No new messages"

                    elif function_name == "getEmailAttachments":

                        print(f'\nGet email attachments function called')

                        attachment = [Attachment(data) for data in result.get('value')]

                        for data in result.get('value'):
                            self.workingEmail.assignAttachment(data)

                        print(self.workingEmail)

                        NumberOfAttachments = len(self.workingEmail.attachments)

                        for item in attachment:

                            print(f'attachment id {item.id}')
                            print(f'attachment.name {item.name}')
                            print(f'attachment.content_type {item.content_type}')
                            print(f'attachment.size {item.size}')
                            print(f'attachment.is_inline {item.is_inline}')
                            print(f'attachment.content_id {item.content_id}')
                            print(f'attachment.content_location{item.content_location}')

                            splunkit(f'Processing attachment: {item.name}', "info")
                            self.logger.info(f"Processing attachment: {item.name}")

                            if item.content_type == 'application/pdf':

                                text, tables, coordinates, md_text = pymuLLM(item.name, item.content_bytes)

                                if not md_text or not "".join(text).strip():
                                    raise Exception("PDF extracted as blank")

                                try:
                                    classification = classifyAttachment(md_text)

                                    if classification == "max_new_token_error":
                                        raise Exception("max_new_token_error")

                                    if not classification:
                                        splunkit(f'{item.name} identified as non-invoice', "info")
                                        self.logger.info(f"{item.name} identified as non-invoice")
                                        continue

                                except Exception as e:
                                    self.RTB_reason = "llama32 max_new_token_error"
                                    splunkit("llama32 max_new_token_error", "error")
                                    self.logger.info(f"llama32 error: {e}")
                                    draft_message = CDR_unableToProcessMaxTokens_prompt
                                    createReplyMessage(self.workingEmail.id, draft_message)
                                    return "max_new_token_error"

                                try:
                                    voucher_dict, evidence_dict = extract_invoice_from_pdf_attachment(
                                        item.content_bytes
                                    )
                                except Exception as e:
                                    self.RTB_reason = f"pdf_vision extract failed: {e}"
                                    splunkit(self.RTB_reason, "error")
                                    self.logger.info(self.RTB_reason)
                                    draft_message = CDR_UnreadableAttachment_prompt
                                    createReplyMessage(self.workingEmail.id, draft_message)
                                    return "Error"

                                jnewVoucher = voucher_dict
                                extracted_invoice_items = jnewVoucher.pop("invoice_items", [])
                                jnewVoucher["invoice_items"] = []

                                if "%" in (jnewVoucher.get("invoice_tax") or ""):
                                    jnewVoucher["invoice_tax"] = ""

                                newVoucher = [json.dumps(jnewVoucher)]

                                messages = [{
                                    "role": "user",
                                    "content": [{
                                        "type": "text",
                                        "text": body_prompt_text.format(
                                            bodyTxt=self.bodyTxt,
                                            voucher=newVoucher
                                        )
                                    }]
                                }]

                                try:
                                    llamaresult = llama32(messages)
                                    if llamaresult == "max_new_token_error":
                                        raise Exception("max_new_token_error")
                                except Exception as e:
                                    self.RTB_reason = "llama32 max_new_token_error"
                                    splunkit("llama32 max_new_token_error", "error")
                                    self.logger.info(f"llama32 error: {e}")
                                    draft_message = CDR_unableToProcessMaxTokens_prompt
                                    createReplyMessage(self.workingEmail.id, draft_message)
                                    return "max_new_token_error"

                                newVoucher = extract_all_strings_regex(llamaresult)
                                for s in newVoucher:
                                    jnewVoucher = json.loads(s)

                                jnewVoucher = self.parseAndFormatVoucher(jnewVoucher)
                                newVoucher = [json.dumps(jnewVoucher)]

                                json_voucher = validateVoucher(newVoucher)

                            elif item.content_type == 'text/plain':
                                pdfText = item.content_bytes.decode('utf-8')
                                print(f'attachment.content_bytes {pdfText}')
                                continue

                            elif item.content_type == 'image/png' or item.content_type == 'image/jpeg':

                                classifyImageInvoice = checkImageforInvoice(item.content_type, item.content_bytes)
                                print(f'The image is an invoice: {classifyImageInvoice}')
                                self.logger.info(f"The image is an invoice: {classifyImageInvoice}")
                                splunkit(f"{self.workingEmail.id} - The image is an invoice: - {classifyImageInvoice}", "info")

                                if not classifyImageInvoice:
                                    continue

                                messages = [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": image_prompt_text.format(bodyTxt=self.bodyTxt)
                                            },
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:{item.content_type};base64,{item.content_bytes}"
                                                }
                                            }
                                        ]
                                    }
                                ]

                                self.PS_BC_AI_RPT_LOG_DF.at[0,'ERROR_MSG_TXT'] = "Step 4: getEmailAttachments - Success"
                                self.PS_BC_AI_RPT_LOG_DF.at[0,"BC_REMARKS"] = f"Successfuly read {NumberOfAttachments} attachments from the email"
                                self.logger.info(f"End of extracting images: PS_BC_AI_RPT_LOG_DF: {self.PS_BC_AI_RPT_LOG_DF}")
                                splunkit(f"End of extracting images: PS_BC_AI_RPT_LOG_DF: {self.PS_BC_AI_RPT_LOG_DF}", "info")

                                print(f'\nmessage {messages}\n')

                                try:
                                    llamarresult = llama32(messages)
                                    if llamarresult == "max_new_token_error":
                                        raise Exception("max_new_token_error")
                                except Exception as e:
                                    self.RTB_reason = "llama32 max_new_token_error"
                                    splunkit(f"llama32 max_new_token_error", "error")
                                    self.logger.info(f"llama32 error: when calling the llm: {e}")
                                    draft_message = CDR_unableToProcessMaxTokens_prompt
                                    createReplyMessage(self.workingEmail.id, draft_message)
                                    return "max_new_token_error"

                                print(f"\nLlama returned: {llamarresult}.")

                                pixtralresult = pixtral(messages)
                                print(f"\nPixTral returned: {pixtralresult}.")

                                if llamarresult.startswith("The image is not an invoice") and pixtralresult.startswith("The image is not an invoice"):
                                    self.PS_BC_AI_RPT_LOG_DF.at[0,'ERROR_MSG_TXT'] = "Step 4: getEmailAttachments - LLM reported image is not an invoice"
                                    self.PS_BC_AI_RPT_LOG_DF.at[0,"BC_REMARKS"] = f"Successfuly read {NumberOfAttachments} attachments from the email"
                                    self.logger.info(f"End of extracting images: PS_BC_AI_RPT_LOG_DF: {self.PS_BC_AI_RPT_LOG_DF}")
                                    splunkit(f"End of extracting images: PS_BC_AI_RPT_LOG_DF: {self.PS_BC_AI_RPT_LOG_DF}", "info")
                                    self.logger.info(f"Image is not an invoice")
                                    splunkit(f"Image is not an invoice: {self.workingEmail.id}", "info")
                                    self.messages.append({"role": "assistant", "content": "The image is not an invoice"})
                                    continue

                                newVoucher = extract_all_strings_regex(pixtralresult)

                                for x in newVoucher:
                                    jnewVoucher = json.loads(x)

                                extracted_invoice_items = jnewVoucher.pop("invoice_items")
                                jnewVoucher["invoice_items"] = []

                                if "%" in jnewVoucher["invoice_tax"]:
                                    jnewVoucher["invoice_tax"] = ""

                                newVoucher = [json.dumps(jnewVoucher)]

                                messages = [{
                                    "role": "user",
                                    "content": [{
                                        "type": "text",
                                        "text": body_prompt_text.format(bodyTxt=self.bodyTxt, voucher=newVoucher)
                                    }]
                                }]

                                try:
                                    llamaresult = llama32(messages)
                                    if llamaresult == "max_new_token_error":
                                        raise Exception("max_new_token_error")
                                except Exception as e:
                                    self.RTB_reason = "llama32 max_new_token_error"
                                    splunkit(f"llama32 max_new_token_error", "error")
                                    self.logger.info(f"llama32 error: when calling the llm: {e}")
                                    draft_message = CDR_unableToProcessMaxTokens_prompt
                                    draft_message = draft_message.replace("[Reason]", str(e))
                                    createReplyMessage(self.workingEmail.id, draft_message)
                                    return "max_new_token_error"

                                newVoucher = extract_all_strings_regex(llamaresult)

                                for x in newVoucher:
                                    jnewVoucher = json.loads(x)

                                jnewVoucher = self.parseAndFormatVoucher(jnewVoucher)
                                newVoucher = [json.dumps(jnewVoucher)]

                                json_voucher = validateVoucher(newVoucher)

                            elif item.content_type.startswith('application/vnd.openxmlformats-officedocument') or item.content_type.startswith('application/vnd.ms-excel'):

                                text, tables, coordinates, md_text = pymuLLM(item.name, item.content_bytes)
                                if not text or not "".join(text).strip():
                                    raise Exception("Doc extracted as blank")

                                print(f'attachment.content_bytes {md_text}')

                                try:
                                    classification = classifyAttachment(md_text)
                                    if classification == "max_new_token_error":
                                        raise Exception('max_new_token_error')

                                    if not classification:
                                        print(f'{item.name} identified as non-invoice')
                                        splunkit(f'{item.name} identified as non-invoice', "info")
                                        self.logger.info(f"{item.name} identified as non-invoice")
                                        continue

                                except Exception as e:
                                    self.RTB_reason = "llama32 max_new_token_error"
                                    splunkit(f"llama32 max_new_token_error", "error")
                                    self.logger.info(f"llama32 error: when calling the llm: {e}")
                                    draft_message = CDR_unableToProcessMaxTokens_prompt
                                    createReplyMessage(self.workingEmail.id, draft_message)
                                    return "max_new_token_error"

                                messages = [{
                                    "role": "user",
                                    "content": [{
                                        "type": "text",
                                        "text": pdf_prompt_text.format(text=text, tables=tables, coordinates=coordinates)
                                    }]
                                }]

                                self.PS_BC_AI_RPT_LOG_DF.at[0,'ERROR_MSG_TXT'] = "Step 4: getEmailAttachments - Success"
                                self.PS_BC_AI_RPT_LOG_DF.at[0,"BC_REMARKS"] = f"Successfuly read {NumberOfAttachments} attachments from the email"
                                self.logger.info(f"End of extracting word document: PS_BC_AI_RPT_LOG_DF: {self.PS_BC_AI_RPT_LOG_DF}")
                                splunkit(f"End of extracting word document: PS_BC_AI_RPT_LOG_DF: {self.PS_BC_AI_RPT_LOG_DF}", "info")

                                print(f'\nmessage {messages}\n')

                                try:
                                    llamarresult = llama32(messages)
                                    if llamarresult == "max_new_token_error":
                                        raise Exception("max_new_token_error")
                                except Exception as e:
                                    self.RTB_reason = "llama32 max_new_token_error"
                                    splunkit(f"llama32 max_new_token_error", "error")
                                    self.logger.info(f"llama32 error: when calling the llm: {e}")
                                    draft_message = CDR_unableToProcessMaxTokens_prompt
                                    createReplyMessage(self.workingEmail.id, draft_message)
                                    return "max_new_token_error"

                                newVoucher = extract_all_strings_regex(llamarresult)

                                for x in newVoucher:
                                    jnewVoucher = json.loads(x)

                                extracted_invoice_items = jnewVoucher.pop("invoice_items")
                                jnewVoucher["invoice_items"] = []

                                if "%" in jnewVoucher["invoice_tax"]:
                                    jnewVoucher["invoice_tax"] = ""

                                newVoucher = [json.dumps(jnewVoucher)]

                                messages = [{
                                    "role": "user",
                                    "content": [{
                                        "type": "text",
                                        "text": body_prompt_text.format(bodyTxt=self.bodyTxt, voucher=newVoucher)
                                    }]
                                }]

                                try:
                                    llamaresult = llama32(messages)
                                    if llamaresult == "max_new_token_error":
                                        raise Exception("max_new_token_error")
                                except Exception as e:
                                    self.RTB_reason = "llama32 max_new_token_error"
                                    splunkit(f"llama32 max_new_token_error", "error")
                                    self.logger.info(f"llama32 error: when calling the llm: {e}")
                                    draft_message = CDR_unableToProcessMaxTokens_prompt
                                    draft_message = draft_message.replace("[Reason]", str(e))
                                    createReplyMessage(self.workingEmail.id, draft_message)
                                    return "max_new_token_error"

                                newVoucher = extract_all_strings_regex(llamaresult)

                                for x in newVoucher:
                                    jnewVoucher = json.loads(x)

                                jnewVoucher = self.parseAndFormatVoucher(jnewVoucher)
                                newVoucher = [json.dumps(jnewVoucher)]

                                json_voucher = validateVoucher(newVoucher)

                            elif item.content_type == 'application/octet-stream':

                                if item.name.lower().endswith('.pdf'):

                                    text, tables, coordinates, md_text = pymuLLM(item.name, item.content_bytes)
                                    if not md_text or not "".join(text).strip():
                                        raise Exception("PDF extracted as blank")

                                    print(f'attachment.content_bytes {md_text}')

                                    try:
                                        classification = classifyAttachment(md_text)

                                        if classification == "max_new_token_error":
                                            raise Exception("max_new_token_error")

                                        if not classification:
                                            print(f'{item.name} identified as non-invoice')
                                            splunkit(f'{item.name} identified as non-invoice', "info")
                                            self.logger.info(f"{item.name} identified as non-invoice")
                                            continue

                                    except Exception as e:
                                        self.RTB_reason = "llama32 max_new_token_error"
                                        splunkit(f"llama32 max_new_token_error", "error")
                                        self.logger.info(f"llama32 error: when calling the llm: {e}")
                                        draft_message = CDR_unableToProcessMaxTokens_prompt
                                        createReplyMessage(self.workingEmail.id, draft_message)
                                        return "max_new_token_error"

                                    try:
                                        voucher_dict, evidence_dict = extract_invoice_from_pdf_attachment(
                                            item.content_bytes
                                        )
                                    except Exception as e:
                                        self.RTB_reason = f"pdf_vision extract failed: {e}"
                                        splunkit(self.RTB_reason, "error")
                                        self.logger.info(self.RTB_reason)
                                        draft_message = CDR_UnreadableAttachment_prompt
                                        createReplyMessage(self.workingEmail.id, draft_message)
                                        return "Error"

                                    jnewVoucher = voucher_dict
                                    extracted_invoice_items = jnewVoucher.pop("invoice_items", [])
                                    jnewVoucher["invoice_items"] = []

                                    if "%" in (jnewVoucher.get("invoice_tax") or ""):
                                        jnewVoucher["invoice_tax"] = ""

                                    newVoucher = [json.dumps(jnewVoucher)]

                                    messages = [{
                                        "role": "user",
                                        "content": [{
                                            "type": "text",
                                            "text": body_prompt_text.format(
                                                bodyTxt=self.bodyTxt,
                                                voucher=newVoucher
                                            )
                                        }]
                                    }]

                                    try:
                                        llamaresult = llama32(messages)
                                        if llamaresult == "max_new_token_error":
                                            raise Exception("max_new_token_error")
                                    except Exception as e:
                                        self.RTB_reason = "llama32 max_new_token_error"
                                        splunkit("llama32 max_new_token_error", "error")
                                        self.logger.info(f"llama32 error: {e}")
                                        draft_message = CDR_unableToProcessMaxTokens_prompt
                                        createReplyMessage(self.workingEmail.id, draft_message)
                                        return "max_new_token_error"

                                    newVoucher = extract_all_strings_regex(llamaresult)
                                    for s in newVoucher:
                                        jnewVoucher = json.loads(s)

                                    jnewVoucher = self.parseAndFormatVoucher(jnewVoucher)
                                    newVoucher = [json.dumps(jnewVoucher)]
                                    json_voucher = validateVoucher(newVoucher)

                                elif item.name.lower().endswith('.docx') or item.name.lower().endswith('.xls') or item.name.lower().endswith('.xlsx'):
                                    # keep your existing doc logic by reusing pymuLLM + pdf_prompt_text
                                    text, tables, coordinates, md_text = pymuLLM(item.name, item.content_bytes)
                                    if not text or not "".join(text).strip():
                                        raise Exception("Doc extracted as blank")

                                    print(f'attachment.content_bytes {md_text}')

                                    try:
                                        classification = classifyAttachment(md_text)
                                        if classification == "max_new_token_error":
                                            raise Exception('max_new_token_error')
                                        if not classification:
                                            print(f'{item.name} identified as non-invoice')
                                            splunkit(f'{item.name} identified as non-invoice', "info")
                                            self.logger.info(f"{item.name} identified as non-invoice")
                                            continue

                                    except Exception as e:
                                        self.RTB_reason = "llama32 max_new_token_error"
                                        splunkit(f"llama32 max_new_token_error", "error")
                                        self.logger.info(f"llama32 error: when calling the llm: {e}")
                                        draft_message = CDR_unableToProcessMaxTokens_prompt
                                        createReplyMessage(self.workingEmail.id, draft_message)
                                        return "max_new_token_error"

                                    messages = [{
                                        "role": "user",
                                        "content": [{
                                            "type": "text",
                                            "text": pdf_prompt_text.format(text=text, tables=tables, coordinates=coordinates)
                                        }]
                                    }]

                                    print(f'\nmessage {messages}\n')

                                    try:
                                        llamarresult = llama32(messages)
                                        if llamarresult == "max_new_token_error":
                                            raise Exception("max_new_token_error")
                                    except Exception as e:
                                        self.RTB_reason = "llama32 max_new_token_error"
                                        splunkit(f"llama32 max_new_token_error", "error")
                                        self.logger.info(f"llama32 error: when calling the llm: {e}")
                                        draft_message = CDR_unableToProcessMaxTokens_prompt
                                        createReplyMessage(self.workingEmail.id, draft_message)
                                        return "max_new_token_error"

                                    newVoucher = extract_all_strings_regex(llamarresult)

                                    for x in newVoucher:
                                        jnewVoucher = json.loads(x)

                                    extracted_invoice_items = jnewVoucher.pop("invoice_items")
                                    jnewVoucher["invoice_items"] = []

                                    if "%" in jnewVoucher["invoice_tax"]:
                                        jnewVoucher["invoice_tax"] = ""

                                    newVoucher = [json.dumps(jnewVoucher)]

                                    messages = [{
                                        "role": "user",
                                        "content": [{
                                            "type": "text",
                                            "text": body_prompt_text.format(bodyTxt=self.bodyTxt, voucher=newVoucher)
                                        }]
                                    }]

                                    try:
                                        llamaresult = llama32(messages)
                                        if llamaresult == "max_new_token_error":
                                            raise Exception("max_new_token_error")
                                    except Exception as e:
                                        self.RTB_reason = "llama32 max_new_token_error"
                                        splunkit(f"llama32 max_new_token_error", "error")
                                        self.logger.info(f"llama32 error: when calling the llm: {e}")
                                        draft_message = CDR_unableToProcessMaxTokens_prompt
                                        draft_message = draft_message.replace("[Reason]", str(e))
                                        createReplyMessage(self.workingEmail.id, draft_message)
                                        return "max_new_token_error"

                                    newVoucher = extract_all_strings_regex(llamaresult)

                                    for x in newVoucher:
                                        jnewVoucher = json.loads(x)

                                    jnewVoucher = self.parseAndFormatVoucher(jnewVoucher)
                                    newVoucher = [json.dumps(jnewVoucher)]

                                    json_voucher = validateVoucher(newVoucher)

                                else:
                                    print(f'attachment.content_bytes {item.content_bytes}')
                                    continue

                            else:
                                print(f'attachment.content_bytes {item.content_bytes}')
                                continue

                            if json_voucher == "Error":
                                return "Error"

                            if "isvalid" in json_voucher and json_voucher["isvalid"] == "Error":
                                if json_voucher.get("status_code") == 500:
                                    draft_message = CDR_ConnectionError_prompt
                                    self.RTB_reason = "Connection Error"
                                    createReplyMessage(self.workingEmail.id, draft_message)
                                    return False
                                else:
                                    return "Error"

                            if "is_duplicate_invoice" in json_voucher and json_voucher["is_duplicate_invoice"] == "Y":
                                splunkit(
                                    f"Invoice identified as a duplicate from local db lookup: {json_voucher['invoice_number']}, {json_voucher['po_number']}",
                                    "info"
                                )
                                draft_message = (
                                    "We appreciate your submission of invoice [Invoice #] dated [Invoice Date]. "
                                    "However, our system indicates that this invoice is a duplicate of a prior submission. "
                                    "Please verify the invoice to ensure this has not been incorrectly tagged as duplicate. "
                                    "Thank you for your cooperation."
                                )
                                draft_message = draft_message.replace("[Invoice #]", json_voucher["invoice_number"]).replace("[Invoice Date]", json_voucher["invoice_date"])
                                createReplyMessage(self.workingEmail.id, draft_message)
                                self.RTB_reason = "Duplicate Invoice"
                                return False

                            self.messages.append({"role": "assistant", "content": str(json_voucher)})
                            self.logger.info(f'After voucher validation : {json_voucher}')
                            splunkit(f'After voucher validation : {self.workingEmail.id} *** {json_voucher}', "info")

                            try:
                                if json_voucher["isvalid"] == True:

                                    assitantMessage = f'{json_voucher} is valid. Proceeding to next step.'
                                    self.messages.append({"role": "assistant", "content": str(assitantMessage)})
                                    self.logger.info(f'Voucher validation success : {json_voucher["isvalid"]}')
                                    splunkit(f'Voucher validation success : {self.workingEmail.id} *** {json_voucher["isvalid"]}', "info")

                                    self.PS_BC_AI_RPT_LOG_DF.at[0,'BC_REMARKS'] = "Voucher validated successfully"

                                    if json_voucher.get("invoice_number", "") != "":
                                        self.PS_BC_AI_RPT_LOG_DF.at[0,'INVOICE_ID'] = json_voucher["invoice_number"]
                                    if json_voucher.get("invoice_date", "") != "":
                                        self.PS_BC_AI_RPT_LOG_DF.at[0,'INVOICE_DT'] = json_voucher["invoice_date"]
                                    if json_voucher.get("gross_invoice_amount", "") != "":
                                        self.PS_BC_AI_RPT_LOG_DF.at[0,'INVOICE_AMOUNT'] = json_voucher["gross_invoice_amount"].replace(',','').replace('$','')
                                    if json_voucher.get("po_number", "") != "":
                                        self.PS_BC_AI_RPT_LOG_DF.at[0,'PO_ID'] = json_voucher["po_number"]

                                    self.logger.info(f"Return from validateVoucher : PS_BC_AI_RPT_LOG_DF: {self.PS_BC_AI_RPT_LOG_DF}")
                                    splunkit(f"Return from validateVoucher : PS_BC_AI_RPT_LOG_DF: {self.PS_BC_AI_RPT_LOG_DF}", "info")

                                    invoiceItems = json_voucher["invoice_items"]

                                    self.workingEmail.assignVoucher(json_voucher)

                                    result = createDocumentBundle(self.workingEmail, json_voucher, attachment)
                                    updateEmailDisposition(self.PS_BC_AI_RPT_LOG_DF)

                                elif json_voucher["isvalid"] == False:
                                    draft_prompt, reason = triagePeopleSoftInvalid(json_voucher)
                                    self.RTB_reason = reason

                                    if "PO Status Error" in draft_prompt:
                                        subject = "PO Status Error"
                                        body = draft_prompt
                                        toRecipients = "GWProcurement@bcbsfl.com"
                                        createNewMessage(subject, body, toRecipients, None, None, attachment)
                                    elif "Insufficient PO Funding" in draft_prompt:
                                        subject = "Insufficient Funding"
                                        body = draft_prompt
                                        toRecipients = "GWProcurement@bcbsfl.com"
                                        createNewMessage(subject, body, toRecipients, None, None, attachment)
                                    else:
                                        createReplyMessage(self.workingEmail.id, draft_prompt)

                                    self.PS_BC_AI_RPT_LOG_DF.at[0,'ERROR_MSG_TXT'] = "Step 7: Create a draft response - Success"
                                    self.PS_BC_AI_RPT_LOG_DF.at[0,"BC_REMARKS"] = "Draft email for missing information created"
                                    updateEmailDisposition(self.PS_BC_AI_RPT_LOG_DF)

                                    self.workingEmail.assignVoucher(json_voucher)
                                    self.PS_BC_AI_RPT_LOG_DF.at[0,'BC_REMARKS'] = "Voucher validation failed"
                                    updateEmailDisposition(self.PS_BC_AI_RPT_LOG_DF)

                                else:
                                    draft_prompt, reason = triagePeopleSoftInvalid(json_voucher)
                                    self.RTB_reason = reason
                                    createReplyMessage(self.workingEmail.id, draft_prompt)
                                    updateEmailDisposition(self.PS_BC_AI_RPT_LOG_DF)

                            except Exception as e:
                                self.logger.error(f"Error exception trapped after voucher validation: {e}")
                                splunkit(f"Error exception trapped after voucher validation: {e}", "error")
                                updateEmailDisposition(self.PS_BC_AI_RPT_LOG_DF)
                                return "Error"
                            finally:
                                for voucherItem in self.workingEmail.voucher:
                                    if voucherItem.isvalid == True:
                                        result = True
                                    else:
                                        result = False
                                        break

                                print(self.PS_BC_AI_RPT_LOG_DF)

                        if len(self.workingEmail.voucher) == 0:
                            return False
                        return all(v.isvalid is True for v in self.workingEmail.voucher)

                    elif function_name == "validateVoucher":
                        print("\nvalidateVoucher function called")
                        self.messages.append({"role": "assistant", "content": str(result)})
                        print(self.PS_BC_AI_RPT_LOG_DF)

                    elif function_name == "moveMessage":
                        self.PS_BC_AI_RPT_LOG_DF.at[0, "BC_REMARKS"] = "Message moved to " + parameters["folderName"]
                        self.messages.append({"role": "assistant", "content": str(result)})
                        print(self.PS_BC_AI_RPT_LOG_DF)
                        updateEmailDisposition(self.PS_BC_AI_RPT_LOG_DF)

                    elif function_name in ("create_draft", "createReplyMessage"):
                        output = "Draft created."
                        self.messages.append({"role": "assistant", "content": str(output)})
                        self.draft_id = result

                    elif function_name == "updateEmailDisposition":
                        updateEmailDisposition(self.PS_BC_AI_RPT_LOG_DF)
                        self.messages.append({"role": "assistant", "content": str(result)})

                    else:
                        self.messages.append({"role": "assistant", "content": str(result)})
                        print(self.PS_BC_AI_RPT_LOG_DF)

                    return result

                except Exception as e:
                    self.logger.error(f"Error in tool calling routing {function_name}: {e}")
                    splunkit(f"Error in tool calling routing {function_name}: {e}", "error")
                    print(f"\nError in tool calling routing {function_name}: {e}")
                    return "Error"

            self.messages.append({"role": "assistant", "content": str(result)})
            print(self.PS_BC_AI_RPT_LOG_DF)
            return result

        except Exception as e:
            self.logger.error(f"AgentV2: Exception caught on the main block {str(e)}")
            splunkit(f"AgentV2: Exception caught on the main block {str(e)}", "error")
            return "Error"

    def llama(self):
        result = llama32(self.messages)
        try:
            res = json.loads(result.split("<|python_tag|>")[-1])
            function_name = res["name"]
            parameters = res["parameters"]
            self.logger.info(f"llama call return -> function name : {function_name} and parameters : {parameters}")
            splunkit(f"llama call return -> function name : {function_name} and parameters : {parameters}", "info")
            return {"function_name": function_name, "parameters": parameters}
        except Exception as e:
            self.logger.info(f"Agentv2: llama: llama call non function call return -> {e}")
            splunkit(f"Agentv2: llama: llama call non function call return -> {e}", "info")
            print(f"exception trapped {e}")
            return result

    def parseAndFormatVoucher(self, voucher):
        jinvoice_number = voucher["invoice_number"]
        voucher["invoice_number"] = re.sub(r"[^0-9A-Z]", "", jinvoice_number.upper())

        jpo_number = voucher["po_number"]
        match = re.search(r"(?:810|850|816|858|812|817|818|828|830|856)(\d{7})", jpo_number)

        if match:
            voucher["po_number"] = match.group(0)
            splunkit(f"PO number was fixed  {self.workingEmail.id} ", "info")
        else:
            splunkit(f"PO number was not fixed for it was good {self.workingEmail.id} ", "info")

        jinvoice_date = voucher["invoice_date"]
        if jinvoice_date != "":
            date_obj = parse(jinvoice_date)
            voucher["invoice_date"] = date_obj.strftime("%Y-%m-%d")

        voucher["gross_invoice_amount"] = voucher["gross_invoice_amount"].replace("$", "").replace(",", "")
        voucher["invoice_tax"] = voucher["invoice_tax"].replace("$", "").replace(",", "")
        voucher["invoice_freight"] = voucher["invoice_freight"].replace("$", "").replace(",", "")

        for item in voucher["invoice_items"]:
            item["item_unit_price"] = item["item_unit_price"].replace("$", "")
            item["item_total"] = item["item_total"].replace("$", "")

        if "%" in voucher["invoice_tax"]:
            voucher["invoice_tax"] = ""

        for item in voucher["invoice_items"]:
            if item["item_quantity"] != "" and item["item_unit_price"] != "":
                quantity = float(item["item_quantity"])
                unit_price = float(item["item_unit_price"])
                item["item_total"] = str(quantity * unit_price)

        for item in voucher["invoice_items"]:
            if item["item_number"] == "" or item["item_total"] == "":
                voucher["invoice_items"] = []
                break

        return voucher


def HelloEmily(agent: Agent):
    result = getMessageList(1, "Inbox")

    if result is None:
        splunkit("Error calling getMessageList.    ", "error")
        return False

    print("\nGet message list function called")
    print(len(result["value"]))

    if len(result["value"]) > 0:
        agent.workingEmail.assign(result)
        mime_msg = getMIMEMessage(agent.workingEmail.id)

        if mime_msg is None:
            splunkit(f"Error calling getMimeMessage.{agent.workingEmail.id}    ", "error")
            return False

        agent.workingEmail.assignMimeMessage(mime_msg)

        with open(r"c:\data\Outputfiles\Emily_mime_messageContent.eml", "wb") as f:
            f.write(agent.workingEmail.mime_message.content)

        with open(r"c:\data\Outputfiles\Emily_mime_messageText.eml", "wb") as f:
            f.write(agent.workingEmail.mime_message.text.encode("utf-8"))

        agent.bodyTxt = agent.workingEmail.body["content"]
        print(agent.workingEmail)

        # update the log table entries with email dispositions
        agent.PS_BC_AI_RPT_LOG_DF.at[0, "BC_MSG_ID"] = agent.workingEmail.id
        format_date_time = datetime.fromisoformat(agent.workingEmail.received_date_time)
        agent.PS_BC_AI_RPT_LOG_DF.at[0, "EMAIL_DATETIME"] = format_date_time.strftime("%Y-%m-%d %H:%M:%S")
        agent.PS_BC_AI_RPT_LOG_DF["EXPORT_DATE"] = datetime.now().strftime("%Y-%m-%d")
        agent.PS_BC_AI_RPT_LOG_DF.at[0, "BC_EMAIL_SUBJTEXT"] = agent.workingEmail.subject
        agent.PS_BC_AI_RPT_LOG_DF.at[0, "ERROR_MSG_TXT"] = "Step 1: getMessageList - Success"
        agent.PS_BC_AI_RPT_LOG_DF.at[0, "BC_REMARKS"] = "Successfuly read the email from the inbox"

        agent.logger.info(f"PS_BC_AI_RPT_LOG_DF: {agent.PS_BC_AI_RPT_LOG_DF}")
        splunkit(f"PS_BC_AI_RPT_LOG_DF: {agent.PS_BC_AI_RPT_LOG_DF}", "info")
        updateEmailDisposition(agent.PS_BC_AI_RPT_LOG_DF)

        print(agent.PS_BC_AI_RPT_LOG_DF)

        agent.messages.append(
            {
                "role": "assistant",
                "content": "getMessage list successfully fetched the first email from Inbox. PS_BC_AI_RPT_LOG_DF is also updated.",
            }
        )
        agent.messages.append({"role": "assistant", "content": str(result)})
        agent.messages.append(
            {
                "role": "assistant",
                "content": f"you dont have to invoke getMessageList function again. From now onwards you can use the message id {agent.workingEmail.id} to process any other tasks.",
            }
        )
    else:
        agent.messages.append({"role": "assistant", "content": str(result)})
        agent.messages.append({"role": "assistant", "content": "There are no emails in the Inbox. Please check the Inbox."})
        agent.logger.info("There are no emails in the Inbox. Will try again in a few minutes.")
        return True

    if "rush" in str(agent.workingEmail.subject).lower() or "rush" in str(agent.workingEmail.body).lower():
        moveMessage(agent.workingEmail.id, "Rush")

        agent.PS_BC_AI_RPT_LOG_DF.at[0, "ERROR_MSG_TXT"] = "Step 2: Rush Email identified - Success"
        agent.PS_BC_AI_RPT_LOG_DF.at[0, "BC_REMARKS"] = "Email moved to  Rush folder"
        agent.messages.append({"role": "assistant", "content": "Message moved to Rush folder"})

        print(agent.PS_BC_AI_RPT_LOG_DF)
        updateEmailDisposition(agent.PS_BC_AI_RPT_LOG_DF)
        return True

    message = (
        'Task: Determine if the email is invoice related or not? Answer Yes or No in the following format with reason. '
        '{"Ans": , "Reason":} If the email is for a invoice or payment status answer No. '
        "Make sure its properly formatted in json format with necessary double quotes etc. "
    )
    step4response = agent(message)

    print(step4response)

    if step4response == "Error":
        agent.logger.error("HelloEmily: Possible connection error in step 4: Check invoice related email or not")
        splunkit("HelloEmily: Possible connection error in step 4 : Check invoice related email or not", "error")
        agent.PS_BC_AI_RPT_LOG_DF.at[0, "ERROR_MSG_TXT"] = "Step 4: Check invoice related email or not - Failed"
        agent.PS_BC_AI_RPT_LOG_DF.at[0, "BC_REMARKS"] = "Possible llm connection error in step 4 : Check invoice related email or not"
        updateEmailDisposition(agent.PS_BC_AI_RPT_LOG_DF)
        return False

    # elif json.loads(step4response)['Ans'] == 'No':
    if "Classification: Not New Invoice" in step4response or '"Ans": "No"' in step4response:
        message = check_again_prompt
        emailTypeResponse = agent(message)

        if '"Ans": "Yes"' in emailTypeResponse:
            agent.PS_BC_AI_RPT_LOG_DF.at[0, "ERROR_MSG_TXT"] = "Step 4: Vendor Update related email identified - Success"
            agent.PS_BC_AI_RPT_LOG_DF.at[0, "BC_REMARKS"] = "Email identified as Vendor Update related"
            updateEmailDisposition(agent.PS_BC_AI_RPT_LOG_DF)

            draft_message = CDR_VendorUpdate_prompt
            createReplyMessage(agent.workingEmail.id, draft_message)

            moveMessage(agent.workingEmail.id, "Return to Business Vendor")
            agent.PS_BC_AI_RPT_LOG_DF.at[0, "ERROR_MSG_TXT"] = "Step 4: Email moved to Return to Business Vendor folder due to Vendor Update related email - Success"
            agent.PS_BC_AI_RPT_LOG_DF.at[0, "BC_REMARKS"] = "Return to Business Vendor folder"
            updateEmailDisposition(agent.PS_BC_AI_RPT_LOG_DF)

            splunkit(f" {agent.workingEmail} - Vendor Update Related - Move the message to Return to Business Vendor folder - Passed", "info")
            return True

        agent.PS_BC_AI_RPT_LOG_DF.at[0, "ERROR_MSG_TXT"] = "Step 4: Non invoice related email identified - Success"
        agent.PS_BC_AI_RPT_LOG_DF.at[0, "BC_REMARKS"] = "Email identified as non invoice related"
        updateEmailDisposition(agent.PS_BC_AI_RPT_LOG_DF)

        moveMessage(agent.workingEmail.id, "Non-Invoice Related")
        agent.logger.info(f" {agent.workingEmail} - Non-Invoice Related - Move the message to Non-Invoice Related folder - Passed")
        splunkit(f" {agent.workingEmail} - Non-Invoice Related - Move the message to Non-Invoice Related folder - Passed", "info")

        agent.PS_BC_AI_RPT_LOG_DF.at[0, "ERROR_MSG_TXT"] = "Step 4: Email moved to Non-Invoice Related folder - Success"
        agent.PS_BC_AI_RPT_LOG_DF.at[0, "BC_REMARKS"] = "Email moved to Non-Invoice Related folder"
        updateEmailDisposition(agent.PS_BC_AI_RPT_LOG_DF)
        return True

    if "Classification: New Invoice" in step4response or '"Ans": "Yes"' in step4response:
        agent.PS_BC_AI_RPT_LOG_DF.at[0, "ERROR_MSG_TXT"] = "Step 4: Invoice related email identified - Success"
        agent.PS_BC_AI_RPT_LOG_DF.at[0, "BC_REMARKS"] = "Sending to get and process attachments"
        updateEmailDisposition(agent.PS_BC_AI_RPT_LOG_DF)

        # Ask Emily to get the attachments and process it
        message = get_and_process_attachments_prompt
        step5response = agent(message)
        print(step5response)

        if step5response is True:
            moveMessage(agent.workingEmail.id, "Processed by AI Emily")
            agent.logger.info(f" {agent.workingEmail} - Attachment validation passed - Move the message to Processed by AI Emily folder ")
            splunkit(f" {agent.workingEmail} - Attachment validation passed - Move the message to Processed by AI Emily folder ", "info")

            agent.PS_BC_AI_RPT_LOG_DF.at[0, "ERROR_MSG_TXT"] = "Step 6: Email processed, attachment extracted, voucher created - Success"
            agent.PS_BC_AI_RPT_LOG_DF.at[0, "BC_REMARKS"] = "Email moved to Processed by AI Emily folder"
            updateEmailDisposition(agent.PS_BC_AI_RPT_LOG_DF)
            return True

        if step5response in (False, "max_new_token_error", "Error"):
            if step5response == "Error":
                draft_message = CDR_UnreadableAttachment_prompt
                agent.RTB_reason = "Unreadable Attachment"
                createReplyMessage(agent.workingEmail.id, draft_message)

                agent.PS_BC_AI_RPT_LOG_DF.at[0, "ERROR_MSG_TXT"] = "Step 7: Create a draft response - Success"
                agent.PS_BC_AI_RPT_LOG_DF.at[0, "BC_REMARKS"] = "Draft email for unreadable attachment created"
                updateEmailDisposition(agent.PS_BC_AI_RPT_LOG_DF)
                splunkit(f" {agent.workingEmail} - Step 7: Create a draft response - Success", "info")

            moveMessage(agent.workingEmail.id, "Return to Business Vendor")
            agent.logger.info(f" {agent.workingEmail} - Attachment validation failed - Move the message to Return to Business Vendor folder ")
            splunkit(f" {agent.workingEmail} - Attachment validation failed - Move the message to Return to Business Vendor folder ", "info")

            agent.PS_BC_AI_RPT_LOG_DF.at[0, "ERROR_MSG_TXT"] = f"Step 6: Invoice validation failed due to reason: {agent.RTB_reason} - Returned to business"
            agent.PS_BC_AI_RPT_LOG_DF.at[0, "BC_REMARKS"] = "Return to Business Vendor folder"
            updateEmailDisposition(agent.PS_BC_AI_RPT_LOG_DF)
            return True

        draft_message = CDR_UnreadableAttachment_prompt
        createReplyMessage(agent.workingEmail.id, draft_message)

        moveMessage(agent.workingEmail.id, "Return to Business Vendor")
        agent.logger.info(f" {agent.workingEmail} -Step 5 - Move the message to Return to Business Vendor folder due to unreadable attachment")
        splunkit(f" {agent.workingEmail} -Step 5 - Move the message to Return to Business Vendor folder due to unreadable attachment", "info")

        agent.PS_BC_AI_RPT_LOG_DF.at[0, "ERROR_MSG_TXT"] = "Step 5: Email moved to Return to Business Vendor folder due to unreadable attachment - Success"
        agent.PS_BC_AI_RPT_LOG_DF.at[0, "BC_REMARKS"] = "Return to Business Vendor folder"
        updateEmailDisposition(agent.PS_BC_AI_RPT_LOG_DF)
        return True

    return True


if __name__ == "__main__":
    agent = Agent(system_prompt)  # initialize and instantiate the Agent object

    # Call HelloEmily to operate in unsupervised mode
    HelloEmily(agent)

    agent_response = "Your ask: "
    while True:
        ask = input(agent_response)
        if ask == "bye":
            break
        print("\n-------------------------\nCalling Llama...")
        output = agent(ask)
        print(output)
        agent_response = "Your ask:(Type 'bye' to exit) "

    schedule.every(10).seconds.do(lambda: HelloEmily(agent))
    while True:
        start_time = datetime.now()
        sys.stdout.write("\033[92m")
        print(f"\nstart time: {start_time}")
        sys.stdout.write("\033[0m")

        agent.logger.info(f"\nEmily the Email Agent has started the task. -------------------- start time: {start_time}")
        splunkit(
            f"\nEmily the Email Agent has started the task. -------------------- start time:  + str(start_time)",
            "info",
        )

        HelloEmily(agent)

        schedule.run_pending()
        end_time = datetime.now()

        sys.stdout.write("\033[92m")
        print(f"\nend time: {datetime.now()}")
        print(f"\ntotal time: {end_time - start_time}")
        sys.stdout.write("\033[0m")

        splunkit(
            f"\nEmily the Email Agent has completed the task. -------------------- end time:  + str(end_time)",
            "info",
        )
        splunkit(f"\ntotal time: {end_time - start_time}", "info")
        time.sleep(1)
