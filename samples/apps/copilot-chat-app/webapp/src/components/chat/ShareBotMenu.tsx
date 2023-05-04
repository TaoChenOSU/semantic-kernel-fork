// Copyright (c) Microsoft. All rights reserved.

import { FC, useCallback } from 'react';

import { Button, Menu, MenuItem, MenuList, MenuPopover, MenuTrigger, Tooltip } from '@fluentui/react-components';
import { ArrowDownloadRegular, PeopleTeamAddRegular, ShareRegular } from '@fluentui/react-icons';
import React from 'react';
import { useChat } from '../../libs/useChat';
import { useFile } from '../../libs/useFile';
import { InvitationDialog } from './invitation-dialog/InvitationDialog';

interface ShareBotMenuProps {
    chatId: string;
    chatTitle: string;
}

export const ShareBotMenu: FC<ShareBotMenuProps> = ({ chatId, chatTitle }) => {
    const chat = useChat();
    const { downloadFile } = useFile();
    const [ isGettingInvitationLink, setIsGettingInvitationLink ] = React.useState(false);

    const onDownloadBotClick = useCallback(async () => {
        // TODO: Add a loading indicator
        const content = await chat.downloadBot(chatId);
        downloadFile(
            `chat-history-${chatTitle}-${new Date().toISOString()}.json`,
            JSON.stringify(content),
            'text/json',
        );
    }, [chat, chatId, chatTitle, downloadFile]);

    const onInviteOthersClick = () => {
        setIsGettingInvitationLink(true);
    };

    const onInviteOthersCancel = () => {
        setIsGettingInvitationLink(false);
    };

    return (
        <div>
        <Menu>
            <MenuTrigger disableButtonEnhancement>
                <Tooltip content="Share" relationship="label">
                    <Button icon={<ShareRegular />} appearance="transparent" />
                </Tooltip>
            </MenuTrigger>
            <MenuPopover>
                <MenuList>
                    <MenuItem icon={<ArrowDownloadRegular />} onClick={onDownloadBotClick}>
                        Download your Bot
                    </MenuItem>
                    <MenuItem icon={<PeopleTeamAddRegular />} onClick={onInviteOthersClick}>
                        Invite others to your Bot
                    </MenuItem>
                </MenuList>
            </MenuPopover>
        </Menu>
            {isGettingInvitationLink && <InvitationDialog onCancel={onInviteOthersCancel} chatId={chatId} />}
        </div>
    );
};
