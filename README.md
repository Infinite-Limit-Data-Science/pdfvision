

for item in attachment:

    splunkit(f'Processing attachment: {item.name}', "info")
    self.logger.info(f"Processing attachment: {item.name}")

    # =======================
    # 1) PDF ATTACHMENTS
    # =======================
    if item.content_type == 'application/pdf':

        # Light text extraction for classification only
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

        # ===== NEW: Use pdf_vision module =====
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

        # Build structure identical to old pipeline
        jnewVoucher = voucher_dict
        extracted_invoice_items = jnewVoucher.pop("invoice_items", [])
        jnewVoucher["invoice_items"] = []

        if "%" in (jnewVoucher.get("invoice_tax") or ""):
            jnewVoucher["invoice_tax"] = ""

        newVoucher = [json.dumps(jnewVoucher)]

        # ===== Run body enrichment (unchanged logic) =====
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


