// Copyright (c) Microsoft. All rights reserved.

import { IPublicClientApplication, InteractionStatus } from '@azure/msal-browser';
import { Url } from 'url';
import { TokenHelper } from '../auth/TokenHelper';
import { IChatInvitation } from '../models/ChatInvitation';
import { IChatMessage } from '../models/ChatMessage';
import { IChatSession } from '../models/ChatSession';
import { BaseService } from './BaseService';

export class ChatService extends BaseService {
    public createChatAsync = async (
        userId: string,
        userName: string,
        title: string,
        accessToken: string,
    ): Promise<IChatSession> => {
        const body = {
            userId: userId,
            userName: userName,
            title: title,
        };

        const result = await this.getResponseAsync<IChatSession>(
            {
                commandPath: 'chatSession/create',
                method: 'POST',
                body: body,
            },
            accessToken,
        );

        return result;
    };

    public getChatAsync = async (chatId: string, accessToken: string): Promise<IChatSession> => {
        const result = await this.getResponseAsync<IChatSession>(
            {
                commandPath: `chatSession/getChat/${chatId}`,
                method: 'GET',
            },
            accessToken,
        );

        return result;
    };

    public getAllChatsAsync = async (userId: string, accessToken: string): Promise<IChatSession[]> => {
        const result = await this.getResponseAsync<IChatSession[]>(
            {
                commandPath: `chatSession/getAllChats/${userId}`,
                method: 'GET',
            },
            accessToken,
        );
        return result;
    };

    public getChatMessagesAsync = async (
        chatId: string,
        startIdx: number,
        count: number,
        accessToken: string,
    ): Promise<IChatMessage[]> => {
        const result = await this.getResponseAsync<IChatMessage[]>(
            {
                commandPath: `chatSession/getChatMessages/${chatId}?startIdx=${startIdx}&count=${count}`,
                method: 'GET',
            },
            accessToken,
        );

        return result;
    };

    public editChatAsync = async (chatId: string, title: string, accessToken: string): Promise<any> => {
        const body: IChatSession = {
            id: chatId,
            userId: '',
            title: title,
        };

        const result = await this.getResponseAsync<any>(
            {
                commandPath: `chatSession/edit`,
                method: 'POST',
                body: body,
            },
            accessToken,
        );

        return result;
    };

    public getChatInvitationLinkAsync = async (chatId: string, userId: string, accessToken: string): Promise<Url> => {
        const body: IChatInvitation = {
            userId: userId,
            chatId: chatId,
        }

        const result = await this.getResponseAsync<Url>(
            {
                commandPath: `chatInvitation/invite`,
                method: 'POST',
                body: body,
            },
            accessToken,
        );

        return result;
    };

    public getUserIDFromUserEmailAsync = async (
        userEmail: string,
        instance: IPublicClientApplication,
        inProgress: InteractionStatus
    ): Promise<string> => {
        const token = await TokenHelper.getAccessTokenUsingMsal(inProgress, instance, ["User.ReadBasic.All"]);
        const request = new URL(`/v1.0/users?$search="mail:${userEmail}"&$count=true`, 'https://graph.microsoft.com');
        const response = await fetch(request, {
            method: 'GET',
            headers: {
                ConsistencyLevel: 'eventual',
                Authorization: `Bearer ${token}`,
            },
        });

        if (!response || !response.ok) {
            throw new Error('Failed to get user ID from user email');
        }
         
        const data = await response.json();
        if (data.value.length > 0) {
            return data.value[0].id;
        } else {
            throw new Error('Failed to get user ID from user email');
        }
    };
}
