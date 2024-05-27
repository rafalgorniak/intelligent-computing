import aiohttp
import asyncio
import json
from datetime import datetime, timezone
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
import base64
import logging


class KSeFServiceAPI:
    def __init__(self, logger):
        self.logger = logger
        self.http_client = aiohttp.ClientSession()
        self.url_base = "https://ksef-test.mf.gov.pl"
        self.public_key = """
        -----BEGIN PUBLIC KEY-----
        MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAuWosgHSpi
        RLadA0fQbzshi5TluliZfDsJujPlyYqp6A3qnzS3WmHxtwgO58uTb
        emQ1HCC2qwrMwuJqR6l8tgA4ilBMDbEEtkzgbjkJ6xoEqBptgxivP/
        ovOFYYoAnY6brZhXytCamSvjY9KI0g0McRk24pOueXT0cbb0tlwEEj
        VZ8NveQNKT2c1EEE2cjmW0XB3UlIBqNqiY2rWF86DcuFDTUy+KzSmT
        JTFvU/ENNyLTh5kkDOmB1SY1Zaw9/Q6+a4VJ0urKZPw+61jtzWmucp
        4CO2cfXg9qtF6cxFIrgfbtvLofGQg09Bh7Y6ZA5VfMRDVDYLjvHwDY
        UHg2dPIk0wIDAQAB
-----END PUBLIC KEY-----
        """

    async def authorisation_challenge_async(self, nip):
        try:
            authorisation_challenge = {
                "contextIdentifier": {
                    "type": "onip",
                    "identifier": nip
                }
            }

            async with self.http_client.post(
                    f"{self.url_base}/api/online/Session/AuthorisationChallenge",
                    json=authorisation_challenge
            ) as response:
                response_body = await response.text()
                self.logger.info(f"Request 'AuthorisationChallengeAsync'\nBody: {response_body}")

                response_json = json.loads(response_body)

                if "exception" in response_json:
                    exception_code = response_json["exception"]["exceptionDetailList"][0]["exceptionCode"]
                    exception_description = response_json["exception"]["exceptionDetailList"][0]["exceptionDescription"]
                    self.logger.error(f"Exception Code: {exception_code}, Description: {exception_description}")

                challenge = response_json.get("challenge", "")
                timestamp_str = response_json.get("timestamp")
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")).astimezone(timezone.utc)
                self.logger.info("Request 'AuthorisationChallengeAsync' completed successfully")

                return challenge, timestamp
        except json.JSONDecodeError as ex:
            self.logger.error(f"JsonException: {ex}")
            return "", datetime.utcnow()
        except Exception as ex:
            self.logger.error(f"Exception: {ex}")
            return "", datetime.utcnow()

    async def init_token_async(self, nip, token, challenge, timestamp):
        try:
            challenge_time = int((timestamp - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds() * 1000)
            token_message = f"{token}|{challenge_time}"

            public_key = serialization.load_pem_public_key(self.public_key.encode('utf-8'))
            encrypted_token_message = public_key.encrypt(
                token_message.encode('utf-8'),
                padding.PKCS1v15()
            )

            encrypted_token_message_b64 = base64.b64encode(encrypted_token_message).decode('utf-8')

            init_token = f"""
            <ns3:InitSessionTokenRequest
                xmlns="http://ksef.mf.gov.pl/schema/gtw/svc/online/types/2021/10/01/0001"
                xmlns:ns2="http://ksef.mf.gov.pl/schema/gtw/svc/types/2021/10/01/0001"
                xmlns:ns3="http://ksef.mf.gov.pl/schema/gtw/svc/online/auth/request/2021/10/01/0001"
            >
                <ns3:Context>
                    <Challenge>{challenge}</Challenge>
                    <Identifier xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:type="ns2:SubjectIdentifierByCompanyType">
                        <ns2:Identifier>{nip}</ns2:Identifier>
                    </Identifier>
                    <DocumentType>
                        <ns2:Service>KSeF</ns2:Service>
                        <ns2:FormCode>
                            <ns2:SystemCode>FA (2)</ns2:SystemCode>
                            <ns2:SchemaVersion>1-0E</ns2:SchemaVersion>
                            <ns2:TargetNamespace>http://crd.gov.pl/wzor/2023/06/29/12648/</ns2:TargetNamespace>
                            <ns2:Value>FA</ns2:Value>
                        </ns2:FormCode>
                    </DocumentType>
                    <Token>{encrypted_token_message_b64}</Token>
                </ns3:Context>
            </ns3:InitSessionTokenRequest>"""

            async with self.http_client.post(
                    f"{self.url_base}/api/online/Session/InitToken",
                    data=init_token,
                    headers={"Content-Type": "application/octet-stream"}
            ) as response:
                response_body = await response.text()
                self.logger.info(f"Request 'InitTokenAsync'\nBody: {response_body}")

                response_json = json.loads(response_body)

                if "exception" in response_json:
                    exception_code = response_json["exception"]["exceptionDetailList"][0]["exceptionCode"]
                    exception_description = response_json["exception"]["exceptionDetailList"][0]["exceptionDescription"]
                    self.logger.error(f"Exception Code: {exception_code}, Description: {exception_description}")

                return response_json
        except Exception as ex:
            self.logger.error(f"Exception: {ex}")
            return {}

    async def close(self):
        await self.http_client.close()


async def main():
    token = "CDD067972497CE08373D51C1E6464143E1C9F4853F5159805FF26BFE1C7EDA2C"
    nip = "1111111111"

    logger = logging.getLogger("KSeFLogger")
    logging.basicConfig(level=logging.INFO)
    ksef_service = KSeFServiceAPI(logger)

    authorisation_challenge_output = await ksef_service.authorisation_challenge_async(nip)

    if authorisation_challenge_output[0] == "":
        await ksef_service.close()
        return

    init_token_output = await ksef_service.init_token_async(
        nip,
        token,
        authorisation_challenge_output[0],
        authorisation_challenge_output[1]
    )

    await ksef_service.close()


if __name__ == "__main__":
    asyncio.run(main())
