from mcpdoc.main import create_server
import os

llm_text_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'acp_llms.txt' 
        )
    )


if __name__ == "__main__":

        # Create a server with documentation sources
    server = create_server(
        [
            {
                "name": "ACP Documentation",
                "llms_txt": llm_text_path,
            },
            # You can add multiple documentation sources
            # {
            #     "name": "Another Documentation",
            #     "llms_txt": "https://example.com/llms.txt",
            # },
        ],
        follow_redirects=True,
        timeout=15.0,
        allowed_domains=["*"],
    )

    # Run the server
    server.run(transport="stdio")