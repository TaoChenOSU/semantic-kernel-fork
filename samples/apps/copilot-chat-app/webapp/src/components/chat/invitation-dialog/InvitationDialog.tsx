// Copyright (c) Microsoft. All rights reserved.

import { useMsal } from "@azure/msal-react";
import { Button, DialogActions, DialogBody, DialogContent, DialogSurface, DialogTitle, Divider, Input, Label, Spinner, makeStyles } from "@fluentui/react-components";
import React from "react";
import { Url } from "url";
import { AuthHelper } from "../../../libs/auth/AuthHelper";
import { ChatService } from "../../../libs/services/ChatService";

const useStyles = makeStyles({
    content: {
        display: "flex",
        flexDirection: "column",
        rowGap: "10px",
    },
    divider: {
        marginTop: "20px",
        marginBottom: "20px",
    },
    copy: {
        display: "flex",
        flexDirection: "row",
    },
    copyLabel: {
        display: "flex",
        alignItems: "center",
    },
    copyButton: {
        marginLeft: 'auto', // align to right
    },
});

interface InvitationDialogProps {
    onCancel: () => void;
    chatId: string;
}

export const InvitationDialog: React.FC<InvitationDialogProps> = (props) => {
    const { onCancel, chatId } = props;
    const { instance, inProgress } = useMsal();
    const chatService = new ChatService(process.env.REACT_APP_BACKEND_URI as string);
    const [isGettingInvitationLink, setIsGettingInvitationLink] = React.useState<boolean>(false);
    const [isFormSubmitted, setIsFormSubmitted] = React.useState<boolean>(false);
    const [errorOccurred, setErrorOccurred] = React.useState<boolean>(false);
    const [errorMessage, setErrorMessage] = React.useState<string>("");
    const [invitationLink, setInvitationLink] = React.useState<Url>();
    const [isLinkCopied, setIsLinkCopied] = React.useState<boolean>(false);
    
    const classes = useStyles();

    const handleSubmit = async (ev: React.FormEvent<HTMLFormElement>) => {
        ev.preventDefault();
        setErrorOccurred(false);
        setIsLinkCopied(false);
        setIsFormSubmitted(true);
        setIsGettingInvitationLink(true);
        
        const userEmail = ev.currentTarget.elements.namedItem("user-email-input") as HTMLInputElement;
        try {
            const userId = await chatService.getUserIDFromUserEmailAsync(userEmail.value, instance, inProgress);
            setInvitationLink(await chatService.getChatInvitationLinkAsync(
                chatId,
                userId,
                await AuthHelper.getSKaaSAccessToken(instance)
            ));
        } catch (error : any) {
            setErrorMessage(error.message);
            setErrorOccurred(true);
        }

        setIsGettingInvitationLink(false);
    };

    const copyLink = () => {
        if (!invitationLink) {
            setErrorMessage("No invitation link is available.");
            setErrorOccurred(true);
            return;
        }
        navigator.clipboard.writeText(invitationLink.toString());
        setIsLinkCopied(true);
    };

    return (
        <div>
            <DialogSurface>
                <form onSubmit={ handleSubmit }>
                    <DialogBody>
                        <DialogTitle>Invite others to your bot</DialogTitle>
                        <DialogContent className={classes.content}>
                            <Label required htmlFor={"user-email-input"}>
                                Please enter the email of the user you would like to invite
                            </Label>
                            <Input required type="email" id={"user-email-input"} />
                        </DialogContent>
                        <DialogActions>
                            <Button appearance="secondary" onClick={onCancel}>Cancel</Button>
                            <Button type="submit" appearance="primary">Invite</Button>
                        </DialogActions>
                    </DialogBody>
                </form>
                <Divider className={classes.divider} />
                {errorOccurred && <Label size="large">{ errorMessage }</Label>}
                {!errorOccurred && isFormSubmitted && isGettingInvitationLink && <Spinner size="large" />}
                {!errorOccurred && isFormSubmitted && !isGettingInvitationLink &&
                    <div>
                        <Label size="large">Copy Link</Label>
                        <div className={classes.copy}>
                            <Label className={classes.copyLabel}>
                                The invited user can use this link to join the chat session.
                            </Label>
                            <Button className={classes.copyButton} appearance="primary" onClick={copyLink}>
                                {isLinkCopied ? "Copied" : "Copy"}
                            </Button>
                        </div>
                    </div>
                }
            </DialogSurface>
        </div>
    );
};